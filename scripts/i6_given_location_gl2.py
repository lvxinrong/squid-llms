import argparse
import hashlib
import json
import runpy
from pathlib import Path

from transformers import AutoTokenizer

from squid.generate.chat import build_qwen_chat_prompt


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "gl2"
PROMPT_VERSION = "gl2_strict_span"


def build_strict_span_task(case, target_method, old_text):
    return f"""你是一名 Java 程序修复助手。本轮只生成局部替换片段。

输出契约：
- 只输出能够直接替换[待替换原文]的 Java 代码片段。
- 禁止输出完整类。
- 禁止输出方法声明或方法签名。
- 禁止重复 Buggy 方法中与本次替换无关的部分。
- 不要 Markdown、JSON 或解释。

[格式示例，与当前缺陷无关]
待替换原文：
int value = 1;

正确输出：
int value = 2;

[目标类]
{case["target_class"]}

[目标方法]
{target_method}

[Buggy 方法源码]
{case["buggy_source"]}

[触发测试]
{case["trigger_test"]}

{case["test_source"]}

[待替换原文]
{old_text}

再次提醒：输出将从待替换原文的第一个字符开始替换，不能包含类声明或方法声明。"""


def prepare_strict_instance(
    case_path,
    key_path,
    tokenizer,
    prepare_instance,
):
    prepared = prepare_instance(
        case_path,
        key_path,
        tokenizer,
    )
    case = json.loads(Path(case_path).read_text(encoding="utf-8"))

    task = build_strict_span_task(
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
        "instance_id": case["instance_id"],
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
        "prompt": prompt,
        "input_ids": input_ids,
        "prompt_tokens": prompt_tokens,
        "budget_tokens": budget_tokens,
        "input_ids_sha256": hashlib.sha256(ids_bytes).hexdigest(),
    })
    prepared["case"] = case
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
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

    tokenizer = AutoTokenizer.from_pretrained(
        prepare_namespace["MODEL_DIR"]
    )
    case_root = Path(args.case_dir)
    prepared_instances = []

    for instance_id in baseline_namespace["INSTANCE_IDS"]:
        case_id = instance_id.lower()
        case_path = case_root / f"{case_id}-case.json"
        key_path = case_root / f"{case_id}-key.json"

        prepared = prepare_strict_instance(
            case_path,
            key_path,
            tokenizer,
            prepare_namespace["prepare_instance"],
        )

        preflight = {
            key: value
            for key, value in prepared.items()
            if key not in {"input_ids", "case"}
        }
        preflight.update({
            "run_id": RUN_ID,
            "prompt_version": PROMPT_VERSION,
            "evaluation_mode": "given_location",
            "reference_new_text_in_prompt": False,
        })

        preflight_path = output_dir / (
            f"i6-{case_id}-given-location-preflight.json"
        )
        preflight_path.write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            "preflight",
            instance_id,
            "prompt_tokens=", prepared["prompt_tokens"],
            "budget_tokens=", prepared["budget_tokens"],
            "hash=", prepared["input_ids_sha256"],
        )
        prepared_instances.append(prepared)

    if args.preflight_only:
        return

    for arm, adapter in (
        ("original", None),
        ("adapter", args.adapter),
    ):
        baseline_namespace["run_arm"](
            arm,
            adapter,
            prepared_instances,
            tokenizer,
            output_dir,
        )

    for instance_id in baseline_namespace["INSTANCE_IDS"]:
        case_id = instance_id.lower()

        for arm in ("original", "adapter"):
            path = output_dir / (
                f"i6-{case_id}-given-location-{arm}.json"
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            report["run_id"] = RUN_ID
            report["prompt_version"] = PROMPT_VERSION
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
