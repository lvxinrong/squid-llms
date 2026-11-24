import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.generate.chat import build_qwen_chat_prompt
from squid.generate.engine import generate
from squid.checkpoints.lora import load_lora
from squid.model.lora import replace_linear_with_lora



ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"

def add_lora(model, rank, alpha):
    for parameter in model.parameters():
        parameter.requires_grad = False
    replace_linear_with_lora(
        model,
        rank=rank,
        alpha=alpha,
    )


def validate_edit(case, edit):
    required = {"action", "target_method", "old_text", "new_text"}
    if not required.issubset(edit):
        return {"status": "missing_fields"}

    if edit["action"] != "replace":
        return {"status": "unsupported_action"}

    signature = re.search(
        r"(?:public|protected|private)\s+"
        r"(?:static\s+)?(?:final\s+)?"
        r"[\w$<>\[\],.?]+\s+(\w+)\s*\(",
        case["buggy_source"],
    )
    target_method = signature.group(1)

    if edit["target_method"] != target_method:
        return {"status": "wrong_target_method"}

    old_text = edit["old_text"]
    new_text = edit["new_text"]

    if not old_text or not new_text:
        return {"status": "empty_edit"}
    if old_text == new_text:
        return {"status": "no_op_edit"}

    match_count = case["buggy_source"].count(old_text)
    if match_count == 0:
        return {"status": "match_not_found", "match_count": 0}
    if match_count > 1:
        return {"status": "ambiguous_match", "match_count": match_count}

    return {
        "status": "ready_to_apply",
        "match_count": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--adapter", default=None)

    args = parser.parse_args()

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    signature = re.search(
        r"(?:public|protected|private)\s+"
        r"(?:static\s+)?(?:final\s+)?"
        r"[\w$<>\[\],.?]+\s+(\w+)\s*\(",
        case["buggy_source"],
    )
    target_method = signature.group(1)

    task = f"""你是一名 Java 程序修复助手。请定位缺陷，并输出一个最小局部编辑。

只输出一个 JSON 对象，不要 Markdown 或解释：
{{
  "action": "replace",
  "target_method": "{target_method}",
  "old_text": "输入源码中需要修改的连续原文",
  "new_text": "用于替换 old_text 的最小修复代码"
}}

要求：
- old_text 必须从输入源码中精确复制，不能使用正则表达式。
- old_text 在目标方法中必须只出现一次。
- new_text 必须保留原有语义，只修复缺陷。
- 不要输出完整方法，不要输出 unified diff，不要修改测试。

[目标类]
{case["target_class"]}

[Buggy 方法源码]
{case["buggy_source"]}

[触发测试]
{case["trigger_test"]}

{case["test_source"]}"""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_qwen3_dir(
        MODEL_DIR,
        context_size=8192,
        device=device,
    )

    if args.adapter:
        add_lora(model, rank=8, alpha=16)
        load_lora(model, args.adapter, strict=True)
        model.to(device).eval()

    prompt = build_qwen_chat_prompt(
        tokenizer,
        [],
        task,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    ).to(device)

    max_new_tokens = 1024
    output = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        context_size=8192,
        temperature=0.0,
        eos_id=tokenizer.eos_token_id,
    )
    generated_ids = output[0, input_ids.shape[1]:].tolist()
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    ).strip()

    evaluation = {
        "status": "invalid_json",
    }
    edit = None

    try:
        edit = json.loads(
            response.split("<|im_end|>", 1)[0].strip()
        )
        evaluation = validate_edit(case, edit)
    except (json.JSONDecodeError, AssertionError, AttributeError):
        pass

    report = {
        "instance_id": case["instance_id"],
        "model": "Qwen3-0.6B original",
        "target_method": target_method,
        "output_contract": "structured_local_edit",
        "prompt": prompt,
        "response": response,
        "edit": edit,
        "evaluation": evaluation,
        "generated_tokens": len(generated_ids),
        "finish_reason": (
            "stop"
            if generated_ids
            and generated_ids[-1] == tokenizer.eos_token_id
            else "length"
        ),
        "reference_patch_in_prompt": False,
        "adapter": args.adapter,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== Structured edit response ===\n")
    print(response)
    print("\nevaluation:")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
