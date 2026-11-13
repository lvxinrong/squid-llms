import json
import statistics
import time
from pathlib import Path
import torch
from squid.checkpoints.qwen import (
    load_qwen2_5,
    load_qwen2_5_dir,
)
from squid.device import get_device, memory_snapshot
from squid.generate.chat import build_qwen_chat_prompt
from squid.generate.engine import generate
from squid.generate.speculative import (
    generate_speculative_cached,
)
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter
import argparse
from squid.generate.benchmark import (
    measure_fixed_length_generation,
    synchronize,
)
from squid.quantize.convert import quantize_model_int8


ROOT = Path(__file__).resolve().parents[1]
DRAFT_WEIGHTS = (
    ROOT
    / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
)
TARGET_DIR = (
    ROOT
    / "data/model_pth/qwen/qwen2_5-1_5b-instruct"
)
REPORT_PATH = (
    ROOT / "data/reports/i3-speculative-qwen-bf16.json"
)

MAX_NEW_TOKENS = 32
DRAFT_LOOKAHEAD = 4
MEASURED_RUNS = 3


def timed(function, device):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    elapsed = time.perf_counter() - start
    return result, elapsed


parser = argparse.ArgumentParser()
parser.add_argument(
    "--lookahead",
    type=int,
    default=4,
)
parser.add_argument(
    "--dtype",
    choices=["bf16", "fp32"],
    default="bf16",
)
parser.add_argument(
    "--quantize",
    choices=["none", "int8"],
    default="none",
)



args = parser.parse_args()
dtype = (
    torch.bfloat16
    if args.dtype == "bf16"
    else torch.float32
)
draft_lookahead = args.lookahead


def main():
    device = get_device()
    tokenizer = QwenTokenizerAdapter(str(TARGET_DIR))

    draft = load_qwen2_5(
        DRAFT_WEIGHTS,
        cfg={"context_length": 4096},
        device=device,
    )

    target = load_qwen2_5_dir(
        TARGET_DIR,
        context_size=4096,
        device=device,
    )

    draft = draft.to(dtype=dtype)
    target = target.to(dtype=dtype)

    quantized_counts = {
        "draft": 0,
        "target": 0,
    }

    if args.quantize == "int8":
        quantized_counts["draft"] = len(
            quantize_model_int8(draft)
        )
        quantized_counts["target"] = len(
            quantize_model_int8(target)
        )

    prompt = build_qwen_chat_prompt(
        tokenizer.tokenizer,
        [],
        "用一句话解释什么是投机解码。",
    )
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    timed(
        lambda: measure_fixed_length_generation(
            target,
            idx.clone(),
            max_new_tokens=4,
            context_size=4096,
            use_cache=True,
        ),
        device,
    )
    timed(
        lambda: generate_speculative_cached(
            draft,
            target,
            idx.clone(),
            max_new_tokens=4,
            context_size=4096,
            draft_lookahead=draft_lookahead,
        ),
        device,
    )

    reference_runs = []
    speculative_runs = []
    reference_ttft_runs = []
    speculative_ttft_runs = []
    reference_output = None
    speculative_output = None
    speculative_stats = None

    for _ in range(MEASURED_RUNS):
        reference_result, elapsed = timed(
            lambda: measure_fixed_length_generation(
                target,
                idx.clone(),
                max_new_tokens=MAX_NEW_TOKENS,
                context_size=4096,
                use_cache=True,
            ),
            device,
        )
        reference_runs.append(elapsed)
        reference_output, reference_metrics = reference_result
        reference_ttft_runs.append(
            reference_metrics["ttft_seconds"]
        )

        result, elapsed = timed(
            lambda: generate_speculative_cached(
                draft,
                target,
                idx.clone(),
                max_new_tokens=MAX_NEW_TOKENS,
                context_size=4096,
                draft_lookahead=draft_lookahead,
                return_stats=True,
            ),
            device,
        )
        speculative_output, speculative_stats = result
        speculative_ttft_runs.append(
            speculative_stats["ttft_seconds"]
        )
        speculative_runs.append(elapsed)

    reference_tokens_per_second = [
        MAX_NEW_TOKENS / elapsed
        for elapsed in reference_runs
    ]
    speculative_tokens_per_second = [
        MAX_NEW_TOKENS / elapsed
        for elapsed in speculative_runs
    ]

    reference_median = statistics.median(reference_runs)
    speculative_median = statistics.median(speculative_runs)

    outputs_match = torch.equal(
        reference_output,
        speculative_output,
    )

    report = {
        "experiment": "i3-speculative-qwen-benchmark",
        "draft_model": "Qwen2.5-0.5B-Instruct",
        "target_model": "Qwen2.5-1.5B-Instruct",
        "device": str(device),
        "compute_dtype": str(next(target.parameters()).dtype),
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": MAX_NEW_TOKENS,
        "draft_lookahead": draft_lookahead,
        "warmup_runs": 1,
        "measured_runs": MEASURED_RUNS,
        "reference": {
            "seconds_runs": reference_runs,
            "seconds_median": reference_median,
            "tokens_per_second_runs": reference_tokens_per_second,
            "tokens_per_second_median": statistics.median(
                reference_tokens_per_second
            ),
            "generated_text": tokenizer.decode(
                reference_output[0, len(prompt_ids):].tolist()
            ),
            "ttft_seconds_runs": reference_ttft_runs,
            "ttft_seconds_median": statistics.median(
                reference_ttft_runs
            ),
        },
        "speculative": {
            "seconds_runs": speculative_runs,
            "seconds_median": speculative_median,
            "tokens_per_second_runs": speculative_tokens_per_second,
            "tokens_per_second_median": statistics.median(
                speculative_tokens_per_second
            ),
            "generated_text": tokenizer.decode(
                speculative_output[0, len(prompt_ids):].tolist()
            ),
            "stats": speculative_stats,
            "ttft_seconds_runs": speculative_ttft_runs,
            "ttft_seconds_median": statistics.median(
                speculative_ttft_runs
            ),
        },
        "quantization": args.quantize,
        "quantized_counts": quantized_counts,
        "ttft_ratio_speculative_over_reference": (
                statistics.median(speculative_ttft_runs)
                / statistics.median(reference_ttft_runs)
        ),
        "output_token_ids_match": outputs_match,
        "speedup_reference_over_speculative": (
            reference_median / speculative_median
        ),
        "draft_acceptance_rate": (
            speculative_stats["draft_tokens_accepted"]
            / speculative_stats["draft_tokens_proposed"]
        ),
        "memory_after_reference": memory_snapshot(device),
        "memory_after_speculative": memory_snapshot(device),
    }

    quant_suffix = (
        ""
        if args.quantize == "none"
        else f"-{args.quantize}"
    )

    report_path = (
            ROOT
            / "data/reports"
            / (
                f"i3-speculative-qwen-{args.dtype}"
                f"{quant_suffix}-k{draft_lookahead}.json"
            )
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n投机解码报告已保存到：{report_path}")

    assert outputs_match, "投机输出与 Target 贪心输出不一致"


if __name__ == "__main__":
    main()
