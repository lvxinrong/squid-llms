import argparse
import gc
import json
import runpy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.lora import load_lora
from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device
from squid.generate.engine import generate


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"
CONTEXT_SIZE = 4096
MAX_NEW_TOKENS = 512
INSTANCE_IDS = ("Lang-10", "Lang-26", "Lang-28", "Math-2")


def split_generated_ids(generated_ids, eos_id):
    stopped = bool(
        generated_ids
        and generated_ids[-1] == eos_id
    )
    content_ids = (
        generated_ids[:-1]
        if stopped
        else generated_ids
    )
    finish_reason = "stop" if stopped else "length"
    return content_ids, finish_reason


def make_edit(prepared, new_text):
    edit = {
        "action": "replace",
        "target_method": prepared["target_method"],
        "old_text": prepared["old_text"],
        "new_text": new_text,
    }
    serialized = json.dumps(edit, ensure_ascii=False)
    assert json.loads(serialized)["new_text"] == new_text
    return edit


def generate_token_ids(model, input_ids, eos_id):
    output = generate(
        model,
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        context_size=CONTEXT_SIZE,
        temperature=0.0,
        eos_id=eos_id,
    )
    generated_ids = output[0, input_ids.shape[1]:].tolist()
    content_ids, finish_reason = split_generated_ids(
        generated_ids,
        eos_id,
    )
    return generated_ids, content_ids, finish_reason


def evaluate_candidate(
    case,
    prepared,
    response,
    finish_reason,
    validate_edit,
):
    edit = make_edit(prepared, response)

    if finish_reason == "length":
        evaluation = {"status": "length_limited"}
    elif not response:
        evaluation = {"status": "empty_edit"}
    elif "```" in response:
        evaluation = {"status": "invalid_output"}
    else:
        evaluation = validate_edit(case, edit)

    return edit, evaluation


def load_model(adapter_path, device):
    model = load_qwen3_dir(
        MODEL_DIR,
        context_size=CONTEXT_SIZE,
        device=device,
    )
    model.eval()

    adapter_rank = None
    adapter_alpha = None

    if adapter_path:
        namespace = runpy.run_path(str(
            ROOT / "scripts/i6_structured_edit_baseline.py"
        ))
        adapter_rank, adapter_alpha = namespace[
            "read_adapter_config"
        ](adapter_path)
        namespace["add_lora"](
            model,
            rank=adapter_rank,
            alpha=adapter_alpha,
        )
        load_lora(model, adapter_path, strict=True)

    model.to(device).eval()
    return model, adapter_rank, adapter_alpha


def run_arm(
    arm,
    adapter_path,
    prepared_instances,
    tokenizer,
    output_dir,
):
    device = get_device()
    torch.manual_seed(123)
    model, adapter_rank, adapter_alpha = load_model(
        adapter_path,
        device,
    )

    baseline_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_structured_edit_baseline.py"
    ))
    validate_edit = baseline_namespace["validate_edit"]

    dtype = str(next(model.parameters()).dtype)

    for prepared in prepared_instances:
        input_ids = prepared["input_ids"].to(device)
        generated_ids, content_ids, finish_reason = (
            generate_token_ids(
                model,
                input_ids,
                tokenizer.eos_token_id,
            )
        )
        response = tokenizer.decode(
            content_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        edit, evaluation = evaluate_candidate(
            prepared["case"],
            prepared,
            response,
            finish_reason,
            validate_edit,
        )

        report = {
            "instance_id": prepared["instance_id"],
            "evaluation_mode": "given_location",
            "arm": arm,
            "adapter": adapter_path,
            "adapter_rank": adapter_rank,
            "adapter_alpha": adapter_alpha,
            "prompt": prepared["prompt"],
            "prompt_tokens": prepared["prompt_tokens"],
            "input_ids_sha256": prepared["input_ids_sha256"],
            "context_size": CONTEXT_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "enable_thinking": False,
            "device": str(device),
            "dtype": dtype,
            "generated_ids": generated_ids,
            "generated_tokens": len(generated_ids),
            "content_tokens": len(content_ids),
            "finish_reason": finish_reason,
            "response": response,
            "edit": edit,
            "evaluation": evaluation,
            "reference_new_text_in_prompt": False,
        }

        case_id = prepared["instance_id"].lower()
        report_path = output_dir / (
            f"i6-{case_id}-given-location-{arm}.json"
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            arm,
            prepared["instance_id"],
            evaluation["status"],
            finish_reason,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_given_location_prepare.py"
    ))
    prepare_instance = prepare_namespace["prepare_instance"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    case_root = Path(args.case_dir)
    prepared_instances = []

    for instance_id in INSTANCE_IDS:
        case_id = instance_id.lower()
        case_path = case_root / f"{case_id}-case.json"
        key_path = case_root / f"{case_id}-key.json"

        prepared = prepare_instance(
            case_path,
            key_path,
            tokenizer,
        )
        prepared["case"] = json.loads(
            case_path.read_text(encoding="utf-8")
        )

        preflight_path = output_dir / (
            f"i6-{case_id}-given-location-preflight.json"
        )
        if preflight_path.exists():
            preflight = json.loads(
                preflight_path.read_text(encoding="utf-8")
            )
            assert preflight["input_ids_sha256"] == (
                prepared["input_ids_sha256"]
            )

        prepared_instances.append(prepared)

    run_arm(
        "original",
        None,
        prepared_instances,
        tokenizer,
        output_dir,
    )
    run_arm(
        "adapter",
        args.adapter,
        prepared_instances,
        tokenizer,
        output_dir,
    )


if __name__ == "__main__":
    main()
