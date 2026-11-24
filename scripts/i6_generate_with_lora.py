import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.lora import load_lora
from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.generate.chat import build_qwen_chat_prompt
from squid.generate.engine import generate
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    case = json.loads(
        Path(args.case).read_text(encoding="utf-8")
    )

    signature = re.search(
        r"public static \w+\s+(\w+)\s*\(",
        case["buggy_source"],
    )
    assert signature is not None
    target_method = signature.group(1)

    task = f"""你是一名 Java 程序修复助手。请阅读 buggy 源码和触发测试，定位缺陷，并输出修复后的完整 {target_method} 方法。

要求：
1. 只输出完整的 Java 方法源码，不要输出 unified diff、解释或 Markdown 代码围栏。
2. 不要删除或截断方法，不要改变方法签名。
3. 保留与缺陷无关的原有逻辑，只做最小修复。
4. 修复必须满足触发测试中的类型与格式要求。
5. 不要复述题目。

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

    max_new_tokens = 2048
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

    report = {
        "instance_id": case["instance_id"],
        "data_role": case["data_role"],
        "base_model": "Qwen3-0.6B original",
        "adapter": args.adapter,
        "rank": 8,
        "alpha": 16,
        "target_method": target_method,
        "context_size": 8192,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "enable_thinking": False,
        "output_contract": "full_corrected_method",
        "prompt": prompt,
        "response": response,
        "generated_tokens": len(generated_ids),
        "finish_reason": (
            "stop"
            if generated_ids
            and generated_ids[-1] == tokenizer.eos_token_id
            else "length"
        ),
        "reference_patch_in_prompt": False,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== LoRA model response ===\n")
    print(response)
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()
