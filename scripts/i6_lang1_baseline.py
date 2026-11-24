import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.generate.chat import build_qwen_chat_prompt
from squid.generate.engine import generate
import argparse


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    case_path = Path(args.case)
    report_path = Path(args.report)

    case = json.loads(case_path.read_text(encoding="utf-8"))

    task = f"""你是一名 Java 程序修复助手。请阅读 buggy 源码和触发测试，定位缺陷并给出最小修复 Patch。

要求：
1. 只输出 unified diff，不要输出解释或 Markdown 代码围栏。
2. Patch 必须能够直接应用到当前 buggy 源码。
3. 不要修改测试文件，不要增加无关重构。
4. 定位缺陷并修复它，不要复述题目。

[目标类]
{case["target_class"]}

[源码文件]
{case["source_file"]}

[Buggy 源码片段]
{case["buggy_source"]}

[触发测试]
{case["trigger_test"]}

{case["test_source"]}"""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_qwen3_dir(
        MODEL_DIR,
        context_size=4096,
        device=device,
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
    ).to(device)

    max_new_tokens = 1024
    output = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        context_size=4096,
        temperature=0.0,
        eos_id=tokenizer.eos_token_id,
    )

    generated_ids = output[0, input_ids.shape[1]:].tolist()
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    ).strip()

    finish_reason = (
        "stop"
        if generated_ids
        and generated_ids[-1] == tokenizer.eos_token_id
        else "length"
    )

    report = {
        "instance_id": case["instance_id"],
        "model": "Qwen3-0.6B original",
        "device": str(device),
        "context_size": 4096,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "enable_thinking": False,
        "prompt": prompt,
        "response": response,
        "generated_tokens": len(generated_ids),
        "finish_reason": finish_reason,
        "evaluation_status": "generation_only",
        "reference_patch_in_prompt": False,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== Model response ===\n")
    print(response)
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()

