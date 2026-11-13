"""I3 番外：对照 V4 与官方 Qwen3-0.6B 在相同输入上的 logits。

用途：验收从零实现的显式 head_dim、QK RMSNorm、GQA、RoPE 和独立权重映射。
输入：本地 Qwen3-0.6B 与 tokenizer；输出 data/reports/i3-qwen3-v4-load-probe.json。
边界：当前只验证 CPU / FP32 / 固定输入的前向，不代表 cache decode、对话或所有 Qwen3 配置已通过。
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"
REPORT_PATH = ROOT / "data/reports/i3-qwen3-v4-load-probe.json"


def main():
    device = torch.device("cpu")
    prompt = "The future of artificial intelligence is"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to(device)

    v4_model = load_qwen3_dir(
        MODEL_DIR,
        context_size=256,
        device=device,
    )
    official_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
    ).to(device).eval()

    with torch.no_grad():
        v4_logits = v4_model(input_ids)
        official_logits = official_model(input_ids).logits

    difference = v4_logits - official_logits

    report = {
        "model": "Qwen3-0.6B",
        "device": str(device),
        "dtype": str(v4_logits.dtype),
        "prompt": prompt,
        "prompt_tokens": input_ids.shape[1],
        "v4_shape": list(v4_logits.shape),
        "official_shape": list(official_logits.shape),
        "max_abs_difference": difference.abs().max().item(),
        "mean_abs_difference": difference.abs().mean().item(),
        "all_finite": torch.isfinite(v4_logits).all().item(),
        "top1_match": (
            v4_logits[0, -1].argmax().item()
            == official_logits[0, -1].argmax().item()
        ),
        "allclose_1e_4": torch.allclose(
            v4_logits,
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
    print(f"\nV4 加载对照报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()
