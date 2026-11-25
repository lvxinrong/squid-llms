import argparse
import hashlib
import json
import runpy
from pathlib import Path

from transformers import AutoTokenizer

from squid.generate.chat import build_qwen_chat_prompt


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"
CONTEXT_SIZE = 4096
MAX_NEW_TOKENS = 512


def build_given_location_task(case, target_method, old_text):
    return f"""你是一名 Java 程序修复助手。本轮已给定需要替换的原文，你只需要生成修复后的替换代码。

只输出 Java 代码，不要 Markdown、JSON 或解释。
输出将原样替换给定原文，必须保留必要缩进。
不要修改测试，不要输出与替换无关的代码。

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
{old_text}"""


def prepare_instance(case_path, key_path, tokenizer):
    case = json.loads(Path(case_path).read_text(encoding="utf-8"))
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))

    namespace = runpy.run_path(str(
        ROOT / "scripts/i6_structured_sft_data.py"
    ))
    target_method = namespace["re_target_method"](
        case["buggy_source"]
    )
    reference_edit = namespace["reference_to_edit"](
        key["reference_patch"],
        target_method,
        case["buggy_source"],
    )

    old_text = reference_edit["old_text"]
    assert case["buggy_source"].count(old_text) == 1

    task = build_given_location_task(
        case,
        target_method,
        old_text,
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
    budget_tokens = prompt_tokens + MAX_NEW_TOKENS
    assert budget_tokens <= CONTEXT_SIZE, {
        "instance_id": case["instance_id"],
        "prompt_tokens": prompt_tokens,
        "budget_tokens": budget_tokens,
        "context_size": CONTEXT_SIZE,
    }

    reference_new_text_tokens = len(tokenizer.encode(
        reference_edit["new_text"],
        add_special_tokens=False,
    ))
    assert reference_new_text_tokens + 1 <= MAX_NEW_TOKENS, {
        "instance_id": case["instance_id"],
        "reference_new_text_tokens": reference_new_text_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    ids_bytes = (
        input_ids.detach()
        .cpu()
        .contiguous()
        .numpy()
        .tobytes()
    )

    return {
        "instance_id": case["instance_id"],
        "target_method": target_method,
        "old_text": old_text,
        "prompt": prompt,
        "input_ids": input_ids,
        "prompt_tokens": prompt_tokens,
        "budget_tokens": budget_tokens,
        "input_ids_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        "reference_new_text_tokens": reference_new_text_tokens,
        "reference_with_eos_tokens": reference_new_text_tokens + 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    prepared = prepare_instance(
        args.case,
        args.key,
        tokenizer,
    )

    report = {
        key: value
        for key, value in prepared.items()
        if key != "input_ids"
    }
    report.update({
        "evaluation_mode": "given_location",
        "context_size": CONTEXT_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "enable_thinking": False,
        "reference_new_text_in_prompt": False,
    })

    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "instance_id": report["instance_id"],
        "prompt_tokens": report["prompt_tokens"],
        "budget_tokens": report["budget_tokens"],
        "reference_with_eos_tokens": (
            report["reference_with_eos_tokens"]
        ),
        "input_ids_sha256": report["input_ids_sha256"],
    }, ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
