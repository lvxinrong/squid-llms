"""I3 番外：对照 V4 与官方 Qwen3 的连续 cache decode 和固定生成。

用途：验证 V4 不只 prefill 正确，还能在逐 token 缓存推理与生成中保持行为一致。
输入：本地 Qwen3-0.6B；输出 data/reports/i3-qwen3-v4-generation-probe.json。
边界：当前只覆盖 CPU / FP32 / eager、固定输入和短序列，不代表所有 dtype、设备或 Qwen3 配置已通过。
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.generate.engine import generate


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"
REPORT_PATH = ROOT / "data/reports/i3-qwen3-v4-generation-probe.json"


def compare_logits(v4_logits, official_logits):
    difference = v4_logits - official_logits
    return {
        "max_abs_difference": difference.abs().max().item(),
        "mean_abs_difference": difference.abs().mean().item(),
        "allclose_1e_4": torch.allclose(
            v4_logits,
            official_logits,
            rtol=1e-4,
            atol=1e-4,
        ),
    }


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
        attn_implementation="eager",
    ).to(device).eval()

    cache_steps = []

    with torch.no_grad():
        v4_logits, v4_past = v4_model(
            input_ids,
            use_cache=True,
        )
        official_output = official_model(
            input_ids,
            use_cache=True,
        )
        official_logits = official_output.logits
        official_past = official_output.past_key_values

        for step in range(3):
            v4_next = v4_logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )
            official_next = official_logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            cache_steps.append({
                "step": step + 1,
                "v4_token": v4_next.item(),
                "official_token": official_next.item(),
                "token_match": v4_next.item() == official_next.item(),
                **compare_logits(
                    v4_logits[:, -1, :],
                    official_logits[:, -1, :],
                ),
            })

            v4_logits, v4_past = v4_model(
                v4_next,
                past_kv=v4_past,
                use_cache=True,
            )
            official_output = official_model(
                v4_next,
                past_key_values=official_past,
                use_cache=True,
            )
            official_logits = official_output.logits
            official_past = official_output.past_key_values

    with torch.no_grad():
        v4_output = generate(
            v4_model,
            input_ids.clone(),
            max_new_tokens=8,
            context_size=256,
            temperature=0.0,
            eos_id=tokenizer.eos_token_id,
        )
        official_output = official_model.generate(
            input_ids.clone(),
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=(
                tokenizer.pad_token_id
                or tokenizer.eos_token_id
            ),
        )

    prompt_length = input_ids.shape[1]
    v4_generated = v4_output[0, prompt_length:].tolist()
    official_generated = official_output[0, prompt_length:].tolist()

    report = {
        "model": "Qwen3-0.6B",
        "device": str(device),
        "dtype": "torch.float32",
        "prompt": prompt,
        "cache_steps": cache_steps,
        "all_cache_tokens_match": all(
            step["token_match"] for step in cache_steps
        ),
        "v4_generated_ids": v4_generated,
        "official_generated_ids": official_generated,
        "v4_generated_text": tokenizer.decode(v4_generated),
        "official_generated_text": tokenizer.decode(official_generated),
        "generated_lengths_match": (
            len(v4_generated) == len(official_generated)
        ),
        "generated_tokens_match": v4_generated == official_generated,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nV4 生成对照报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()
