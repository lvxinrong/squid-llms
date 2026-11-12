"""I3-2 W8A16（FP16）生成对照：验证 RMSNorm 修复后 FP16 参考与 INT8 权重的文本和性能。

用途：在 FP16 参考组上单独测量权重量化开销，同时确认此前 RMSNorm 平方溢出已消失。
配置：同一 prompt、32 个新 token、贪心、1 次 warmup + 3 次测量；使用 FP32 累积 RMSNorm。
输出：data/reports/i3-w8a16-generation-mps.json。
边界：速度结果仍属于反量化 + FP16 F.linear 的教学路径。
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
from squid.quantize.convert import quantize_model_int8
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = (
    ROOT
    / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
)
REPORT_PATH = (
    ROOT / "data/reports/i3-w8a16-generation-mps.json"
)

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

    model = model.half()

    reference = run_benchmark(model, tokenizer, idx)

    replaced = quantize_model_int8(model)
    assert len(replaced) == 168

    quantized = run_benchmark(model, tokenizer, idx)

    report = {
        "experiment": "i3-w8a16-generation-benchmark",
        "model": "Qwen2.5-0.5B-Instruct",
        "device": str(device),
        "weight_path": "per-row INT8, out_head excluded",
        "compute_dtype": "torch.float16",
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "warmup_runs": 1,
        "measured_runs": MEASURED_RUNS,
        "reference_fp16": reference,
        "quantized_int8": quantized,
        "generated_text_match": (
            reference["generated_text"] == quantized["generated_text"]
        ),
        "ttft_ratio_quantized_over_fp16": (
            quantized["ttft_seconds_median"]
            / reference["ttft_seconds_median"]
        ),
        "decode_ratio_quantized_over_fp16": (
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
