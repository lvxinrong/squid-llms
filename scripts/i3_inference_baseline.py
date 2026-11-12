"""I3-0 推理基线：用真实 Qwen2.5-0.5B-Instruct 对照 cache / no-cache 固定长度生成。

用途：分别测量 TTFT、首字后的 decode tokens/s、总耗时、KV Cache 实际字节数与设备内存快照。
配置：FP32、固定 prompt、32 个新 token、贪心解码、禁用 EOS 提前停止， warmup 后独立复测。
输出：data/reports/i3-inference-baseline-cache.json 与 no-cache.json。
边界：固定长度输出不代表正式服务的 EOS 行为。
"""

import json
import platform
from pathlib import Path

import torch

from squid.checkpoints.qwen import load_qwen2_5
from squid.device import get_device, memory_snapshot, reset_peak_memory_stats
from squid.generate.benchmark import (
    measure_fixed_length_generation,
    synchronize,
)
from squid.generate.chat import build_qwen_chat_prompt
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
REPORT_DIR = ROOT / "data/reports"
MAX_NEW_TOKENS = 32
CONTEXT_SIZE = 4096


def main():
    device = get_device()
    model = load_qwen2_5(
        WEIGHTS_PATH,
        cfg={"context_length": CONTEXT_SIZE},
        device=device,
    )
    tokenizer = QwenTokenizerAdapter("Qwen/Qwen2.5-0.5B-Instruct")
    prompt = build_qwen_chat_prompt(
        tokenizer.tokenizer,
        [],
        "用一句话解释为什么 KV Cache 能加速文本生成。",
    )
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    memory_after_model_load = memory_snapshot(device)

    outputs = {}
    for use_cache in (True, False):
        mode = "cache" if use_cache else "no-cache"

        measure_fixed_length_generation(
            model,
            idx.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            context_size=CONTEXT_SIZE,
            use_cache=use_cache,
        )

        synchronize(device)
        reset_peak_memory_stats(device)
        output, metrics = measure_fixed_length_generation(
            model,
            idx.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            context_size=CONTEXT_SIZE,
            use_cache=use_cache,
        )
        measured_memory = memory_snapshot(device)
        outputs[mode] = output

        report = {
            "experiment": "i3-inference-baseline",
            "model": "Qwen2.5-0.5B-Instruct",
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "prompt": prompt,
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": MAX_NEW_TOKENS,
            "context_size": CONTEXT_SIZE,
            "temperature": 0.0,
            "top_k": None,
            "eos_early_stop": False,
            "warmup_runs": 1,
            "measured_runs": 1,
            **metrics,
            "kv_cache_bytes_per_prompt_token": (
                metrics["kv_cache_after_prefill_bytes"] / len(prompt_ids)
                if use_cache
                else None
            ),
            "memory_after_model_load": memory_after_model_load,
            "memory_after_measured": measured_memory,
            "generated_text": tokenizer.decode(
                output[0, len(prompt_ids):].tolist(),
            ),
        }

        report_path = REPORT_DIR / f"i3-inference-baseline-{mode}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n{mode} 基线报告已保存到：{report_path}\n")

    outputs_match = torch.equal(outputs["cache"], outputs["no-cache"])
    if not outputs_match:
        raise AssertionError("cache 与 no-cache 的固定长度输出不一致")

    print("验收：cache 与 no-cache 输出逐 token 一致。")


if __name__ == "__main__":
    main()
