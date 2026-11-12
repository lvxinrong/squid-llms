"""I3-2 W4A32 生成对照：比较 FP32 与 packed INT4 权重在真实 Qwen 上的文本和性能。

用途：把解包、符号恢复、反量化和 FP32 计算的全部开销纳入 TTFT / decode 测量，而非只报告压缩倍率。
配置：同一 prompt、32 个新 token、贪心、固定长度、1 次 warmup + 3 次正式测量。
输出：data/reports/i3-int4-generation-mps.json。
边界：当前朴素实现的质量与速度不代表专业 INT4 推理框架。
"""

import json
import statistics
from pathlib import Path

import torch

from squid.checkpoints.qwen import load_qwen2_5
from squid.device import (
    get_device,
    memory_snapshot,
    reset_peak_memory_stats,
)
from squid.generate.benchmark import (
    measure_fixed_length_generation,
    synchronize,
)
from squid.generate.chat import build_qwen_chat_prompt
from squid.quantize.convert import quantize_model_int4
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = (
    ROOT
    / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
)
REPORT_PATH = ROOT / "data/reports/i3-int4-generation-mps.json"
MAX_NEW_TOKENS = 32
MEASURED_RUNS = 3


def run_benchmark(model, tokenizer, idx):
    measure_fixed_length_generation(
        model,
        idx.clone(),
        max_new_tokens=MAX_NEW_TOKENS,
        context_size=4096,
        use_cache=True,
    )

    runs = []
    output = None

    for _ in range(MEASURED_RUNS):
        synchronize(idx.device)
        reset_peak_memory_stats(idx.device)

        output, metrics = measure_fixed_length_generation(
            model,
            idx.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            context_size=4096,
            use_cache=True,
        )
        runs.append(metrics)

    generated_ids = output[0, idx.shape[1]:].tolist()

    return {
        "ttft_seconds_runs": [
            run["ttft_seconds"] for run in runs
        ],
        "decode_tokens_per_second_runs": [
            run["decode_tokens_per_second"] for run in runs
        ],
        "ttft_seconds_median": statistics.median(
            run["ttft_seconds"] for run in runs
        ),
        "decode_tokens_per_second_median": statistics.median(
            run["decode_tokens_per_second"] for run in runs
        ),
        "total_seconds_median": statistics.median(
            run["total_seconds"] for run in runs
        ),
        "generated_text": tokenizer.decode(generated_ids),
        "memory_after_last_run": memory_snapshot(idx.device),
    }


def main():
    device = get_device()
    model = load_qwen2_5(
        WEIGHTS_PATH,
        cfg={"context_length": 4096},
        device=device,
    )
    tokenizer = QwenTokenizerAdapter(
        "Qwen/Qwen2.5-0.5B-Instruct",
    )

    prompt = build_qwen_chat_prompt(
        tokenizer.tokenizer,
        [],
        "用一句话解释什么是权重量化。",
    )
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    reference = run_benchmark(model, tokenizer, idx)

    replaced = quantize_model_int4(model)
    assert len(replaced) == 168

    quantized = run_benchmark(model, tokenizer, idx)

    report = {
        "experiment": "i3-int4-generation-benchmark",
        "model": "Qwen2.5-0.5B-Instruct",
        "device": str(device),
        "weight_path": "per-row INT4 packed, out_head excluded",
        "compute_dtype": "torch.float32",
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "warmup_runs": 1,
        "measured_runs": MEASURED_RUNS,
        "reference_fp32": reference,
        "quantized_int4": quantized,
        "generated_text_match": (
            reference["generated_text"] == quantized["generated_text"]
        ),
        "ttft_ratio_quantized_over_fp32": (
            quantized["ttft_seconds_median"]
            / reference["ttft_seconds_median"]
        ),
        "decode_ratio_quantized_over_fp32": (
            quantized["decode_tokens_per_second_median"]
            / reference["decode_tokens_per_second_median"]
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n生成对照报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()
