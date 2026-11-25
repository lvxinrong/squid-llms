import argparse
import gc
import hashlib
import json
import runpy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.device import get_device
from squid.generate.chat import build_qwen_chat_prompt


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "gl3_fm"
PROMPT_VERSION = "gl3_given_location_full_method"
VAL_DATA = ROOT / "data/instruct/i6_structured_16_val.json"


def build_full_method_task(case, target_method, old_text):
    return f"""你是一名 Java 程序修复助手。本轮已给出目标方法中的待修改原文，请输出修复后的完整目标方法。

输出契约：
- 只输出一个完整 Java 方法，包含方法签名和结束花括号。
- 方法签名必须与输入目标方法一致。
- 不要输出完整类或额外 helper 方法。
- 不要 JSON、Markdown 或解释。
- 输出将用于替换整个目标方法。

[目标类]
{case["target_class"]}

[目标方法]
{target_method}

[Buggy 完整方法]
{case["buggy_source"]}

[触发测试]
{case["trigger_test"]}

{case["test_source"]}

[给定待修改原文]
{old_text}"""


def run_full_method_arm(
    arm,
    adapter_path,
    prepared_instances,
    tokenizer,
    output_dir,
    model_dir,
    dtype,
):
    baseline_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_given_location_baseline.py"
    ))
    strict_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_strict_full_method_apply.py"
    ))

    device = get_device()
    torch.manual_seed(123)
    model, adapter_rank, adapter_alpha = (
        baseline_namespace["load_model"](
            adapter_path,
            device,
            model_dir=model_dir,
            dtype=dtype,
        )
    )
    dtype = str(next(model.parameters()).dtype)

    for prepared in prepared_instances:
        input_ids = prepared["input_ids"].to(device)
        generated_ids, content_ids, finish_reason = (
            baseline_namespace["generate_token_ids"](
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

        evaluation = {"status": "ready_to_apply"}
        validation_error = None
        outside_changes = None

        if finish_reason == "length":
            evaluation = {"status": "length_limited"}
        else:
            try:
                strict_namespace["validate_method_response"](
                    response,
                    prepared["case"]["buggy_source"],
                )
                outside_changes = strict_namespace[
                    "changes_outside_given_old_text"
                ](
                    response,
                    prepared["case"]["buggy_source"],
                    prepared["old_text"],
                )
            except ValueError as error:
                evaluation = {"status": "malformed_method"}
                validation_error = str(error)

        report = {
            "model_dir": str(model_dir),
            "instance_id": prepared["instance_id"],
            "evaluation_mode": "given_location_full_method",
            "run_id": RUN_ID,
            "prompt_version": PROMPT_VERSION,
            "arm": arm,
            "adapter": adapter_path,
            "adapter_rank": adapter_rank,
            "adapter_alpha": adapter_alpha,
            "prompt": prepared["prompt"],
            "prompt_tokens": prepared["prompt_tokens"],
            "input_ids_sha256": prepared["input_ids_sha256"],
            "context_size": 4096,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "enable_thinking": False,
            "device": str(device),
            "dtype": dtype,
            "generated_ids": generated_ids,
            "generated_tokens": len(generated_ids),
            "content_tokens": len(content_ids),
            "finish_reason": finish_reason,
            "response": response,
            "given_old_text": prepared["old_text"],
            "method_changed": (
                response != prepared["case"]["buggy_source"]
            ),
            "changes_outside_given_old_text": outside_changes,
            "evaluation": evaluation,
            "validation_error": validation_error,
            "reference_new_text_in_prompt": False,
            "reference_full_method_in_prompt": False,
        }

        case_id = prepared["instance_id"].lower()
        path = output_dir / (
            f"i6-{case_id}-gl3-fm-{arm}.json"
        )
        path.write_text(
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
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--preflight-only", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_given_location_prepare.py"
    ))
    baseline_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_given_location_baseline.py"
    ))

    val_entries = json.loads(VAL_DATA.read_text(encoding="utf-8"))
    val_by_id = {
        entry["instance_id"]: entry
        for entry in val_entries
    }

    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else prepare_namespace["MODEL_DIR"]
    )
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    case_root = Path(args.case_dir)
    prepared_instances = []

    for instance_id in baseline_namespace["INSTANCE_IDS"]:
        case_id = instance_id.lower()
        case_path = case_root / f"{case_id}-case.json"
        key_path = case_root / f"{case_id}-key.json"

        prepared = prepare_namespace["prepare_instance"](
            case_path,
            key_path,
            tokenizer,
        )
        case = json.loads(case_path.read_text(encoding="utf-8"))
        reference_edit = val_by_id[instance_id]["edit"]

        fixed_method = case["buggy_source"].replace(
            reference_edit["old_text"],
            reference_edit["new_text"],
            1,
        )
        reference_full_tokens = len(tokenizer.encode(
            fixed_method,
            add_special_tokens=False,
        ))
        assert reference_full_tokens + 1 <= 512, {
            "instance_id": instance_id,
            "reference_full_tokens": reference_full_tokens,
        }

        task = build_full_method_task(
            case,
            prepared["target_method"],
            prepared["old_text"],
        )
        prompt = build_qwen_chat_prompt(
            tokenizer,
            [],
            task,
            enable_thinking=False,
        )
        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
        )
        prompt_tokens = input_ids.shape[1]
        budget_tokens = prompt_tokens + 512
        assert budget_tokens <= 4096, {
            "instance_id": instance_id,
            "budget_tokens": budget_tokens,
        }

        ids_bytes = (
            input_ids.detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )
        prepared.update({
            "case": case,
            "prompt": prompt,
            "input_ids": input_ids,
            "prompt_tokens": prompt_tokens,
            "budget_tokens": budget_tokens,
            "input_ids_sha256": hashlib.sha256(
                ids_bytes
            ).hexdigest(),
            "reference_full_method_tokens": (
                reference_full_tokens
            ),
            "reference_full_with_eos_tokens": (
                reference_full_tokens + 1
            ),
        })

        preflight = {
            key: value
            for key, value in prepared.items()
            if key not in {"input_ids", "case"}
        }
        preflight.update({
            "run_id": RUN_ID,
            "prompt_version": PROMPT_VERSION,
            "reference_full_method_in_prompt": False,
        })
        preflight_path = output_dir / (
            f"i6-{case_id}-gl3-fm-preflight.json"
        )
        preflight_path.write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            "preflight",
            instance_id,
            "prompt_tokens=", prompt_tokens,
            "budget_tokens=", budget_tokens,
            "reference_full_with_eos=",
            reference_full_tokens + 1,
            "hash=", prepared["input_ids_sha256"],
        )
        prepared_instances.append(prepared)

    if args.preflight_only:
        return

    run_full_method_arm(
        "original",
        None,
        prepared_instances,
        tokenizer,
        output_dir,
        model_dir,
        dtype,
    )

    if args.adapter:
        run_full_method_arm(
            "adapter",
            args.adapter,
            prepared_instances,
            tokenizer,
            output_dir,
            model_dir,
            dtype,
        )


if __name__ == "__main__":
    main()
