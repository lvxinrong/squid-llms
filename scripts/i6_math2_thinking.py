import argparse
import gc
import hashlib
import json
import runpy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.generate.chat import build_qwen_chat_prompt
from squid.generate.engine import generate


ROOT = Path(__file__).resolve().parents[1]
MAX_NEW_TOKENS = 2048
CONTEXT_SIZE = 4096
INSTANCE_ID = "Math-2"
VAL_DATA = ROOT / "data/instruct/i6_structured_16_val.json"


def extract_final_response(raw_response, thinking):
    if not thinking:
        return raw_response, None

    close_tag = "</think>"
    if raw_response.count(close_tag) != 1:
        raise ValueError(
            "thinking 输出必须且只能包含一个 </think>"
        )

    reasoning, response = raw_response.split(close_tag, 1)
    return response, reasoning


def prepare_thinking_prompt(
    tokenizer,
    task,
    thinking,
):
    prompt = build_qwen_chat_prompt(
        tokenizer,
        [],
        task,
        enable_thinking=thinking,
    )
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    )
    prompt_tokens = input_ids.shape[1]
    budget_tokens = prompt_tokens + MAX_NEW_TOKENS

    assert budget_tokens <= CONTEXT_SIZE, {
        "thinking": thinking,
        "budget_tokens": budget_tokens,
    }

    ids_bytes = (
        input_ids.detach()
        .cpu()
        .contiguous()
        .numpy()
        .tobytes()
    )

    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "prompt_tokens": prompt_tokens,
        "budget_tokens": budget_tokens,
        "input_ids_sha256": hashlib.sha256(
            ids_bytes
        ).hexdigest(),
    }


def evaluate_response(
    raw_response,
    thinking,
    finish_reason,
    case,
    old_text,
    strict_namespace,
):
    result = {
        "evaluation": {"status": "ready_to_apply"},
        "response": raw_response,
        "thinking_closed": None,
        "validation_error": None,
        "method_changed": None,
        "no_op_ignoring_outer_whitespace": None,
        "changes_outside_given_old_text": None,
    }

    if finish_reason == "length":
        result["evaluation"] = {"status": "length_limited"}
        return result

    try:
        response, reasoning = extract_final_response(
            raw_response,
            thinking,
        )
        result["response"] = response
        result["thinking_closed"] = (
            reasoning is not None
        )

        strict_namespace["validate_method_response"](
            response,
            case["buggy_source"],
        )
        result["method_changed"] = (
            response != case["buggy_source"]
        )
        result["no_op_ignoring_outer_whitespace"] = (
            response.strip() == case["buggy_source"].strip()
        )
        result["changes_outside_given_old_text"] = (
            strict_namespace["changes_outside_given_old_text"](
                response,
                case["buggy_source"],
                old_text,
            )
        )
    except ValueError as error:
        result["evaluation"] = {"status": "malformed_method"}
        result["validation_error"] = str(error)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--evidence-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_given_location_prepare.py"
    ))
    full_method_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_given_location_full_method.py"
    ))
    baseline_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_given_location_baseline.py"
    ))
    strict_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_strict_full_method_apply.py"
    ))

    case_id = INSTANCE_ID.lower()
    case_path = Path(args.case_dir) / f"{case_id}-case.json"
    key_path = Path(args.case_dir) / f"{case_id}-key.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    evidence = json.loads(
        Path(args.evidence_file).read_text(encoding="utf-8")
    )[INSTANCE_ID]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prepared = prepare_namespace["prepare_instance"](
        case_path,
        key_path,
        tokenizer,
    )

    val_entries = json.loads(VAL_DATA.read_text(encoding="utf-8"))
    reference_edit = next(
        entry["edit"]
        for entry in val_entries
        if entry["instance_id"] == INSTANCE_ID
    )
    fixed_method = case["buggy_source"].replace(
        reference_edit["old_text"],
        reference_edit["new_text"],
        1,
    )
    reference_full_tokens = len(tokenizer.encode(
        fixed_method,
        add_special_tokens=False,
    ))
    assert reference_full_tokens + 1 <= MAX_NEW_TOKENS

    task = full_method_namespace["build_full_method_task"](
        case,
        prepared["target_method"],
        prepared["old_text"],
        location_lines="2",
        behavior_evidence=evidence,
    )

    prepared_modes = []

    for mode, thinking in (
        ("thinking-off", False),
        ("thinking-on", True),
    ):
        rendered = prepare_thinking_prompt(
            tokenizer,
            task,
            thinking,
        )
        rendered.update({
            "mode": mode,
            "thinking": thinking,
        })
        prepared_modes.append(rendered)

        preflight = {
            key: value
            for key, value in rendered.items()
            if key != "input_ids"
        }
        preflight.update({
            "instance_id": INSTANCE_ID,
            "max_new_tokens": MAX_NEW_TOKENS,
            "reference_full_with_eos_tokens": (
                reference_full_tokens + 1
            ),
            "reference_full_method_in_prompt": False,
        })
        path = output_dir / f"i6-math-2-{mode}-preflight.json"
        path.write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "preflight",
            mode,
            "prompt_tokens=", rendered["prompt_tokens"],
            "budget_tokens=", rendered["budget_tokens"],
            "hash=", rendered["input_ids_sha256"],
        )

    if args.preflight_only:
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = load_qwen3_dir(
        args.model_dir,
        context_size=CONTEXT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval()
    dtype = str(next(model.parameters()).dtype)

    for rendered in prepared_modes:
        torch.manual_seed(123)
        input_ids = rendered["input_ids"].to(device)

        generated_ids, content_ids, finish_reason = (
            baseline_namespace["generate_token_ids"](
                model,
                input_ids,
                tokenizer.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        )
        raw_response = tokenizer.decode(
            content_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        evaluation = evaluate_response(
            raw_response,
            rendered["thinking"],
            finish_reason,
            case,
            prepared["old_text"],
            strict_namespace,
        )

        report = {
            "instance_id": INSTANCE_ID,
            "evaluation_mode": "given_location_full_method",
            "run_id": "math2_thinking_v1",
            "mode": rendered["mode"],
            "enable_thinking": rendered["thinking"],
            "model_dir": args.model_dir,
            "adapter": None,
            "prompt": rendered["prompt"],
            "prompt_tokens": rendered["prompt_tokens"],
            "input_ids_sha256": rendered["input_ids_sha256"],
            "context_size": CONTEXT_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "device": str(device),
            "dtype": dtype,
            "generated_ids": generated_ids,
            "generated_tokens": len(generated_ids),
            "content_tokens": len(content_ids),
            "finish_reason": finish_reason,
            "raw_response": raw_response,
            "given_old_text": prepared["old_text"],
            "reference_full_method_in_prompt": False,
            **evaluation,
        }

        path = output_dir / (
            f"i6-math-2-{rendered['mode']}.json"
        )
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            rendered["mode"],
            evaluation["evaluation"]["status"],
            finish_reason,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
