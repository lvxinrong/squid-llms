"""
同一份 Qwen2.5-1.5B 权重
├─ 用我们的 V3 + 自定义加载器运行
└─ 用 Hugging Face 官方 Qwen2 实现运行
                ↓
输入同一个 prompt
                ↓
比较两边输出的 logits

"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from squid.checkpoints.qwen import load_qwen2_5_dir


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = (
    ROOT
    / "data/model_pth/qwen/qwen2_5-1_5b-instruct"
)
REPORT_PATH = (
    ROOT / "data/reports/i3-qwen1_5b-load-probe.json"
)


def main():
    device = torch.device("cpu")
    prompt = "The future of artificial intelligence is"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to(device)

    squid_model = load_qwen2_5_dir(
        MODEL_DIR,
        context_size=256,
        device=device,
    )

    official_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
    ).to(device).eval()

    with torch.no_grad():
        squid_logits = squid_model(input_ids)
        official_logits = official_model(input_ids).logits

    difference = squid_logits - official_logits

    report = {
        "model": "Qwen2.5-1.5B-Instruct",
        "device": str(device),
        "dtype": str(squid_logits.dtype),
        "prompt": prompt,
        "prompt_tokens": input_ids.shape[1],
        "squid_shape": list(squid_logits.shape),
        "official_shape": list(official_logits.shape),
        "max_abs_difference": difference.abs().max().item(),
        "mean_abs_difference": difference.abs().mean().item(),
        "all_finite": torch.isfinite(squid_logits).all().item(),
        "top1_match": (
            squid_logits[0, -1].argmax().item()
            == official_logits[0, -1].argmax().item()
        ),
        "allclose_1e_4": torch.allclose(
            squid_logits,
            official_logits,
            rtol=1e-4,
            atol=1e-4,
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n加载对照报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()
