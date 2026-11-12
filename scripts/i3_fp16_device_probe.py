"""I3 FP16 排障：用同一批数据对照 CPU / MPS 上 FP32、FP16、BF16 的 loss 与 logits。

用途：先区分设备特定 kernel 与模型 dtype 数值路径，避免把 P16 异常直接归因于 MPS。
输入：真实 Qwen 与单个 128-token heldout batch；输出 data/reports/i3-fp16-device-probe.json。
边界：单批结果只在探针内部对照，不与十批 PPL 混用；CPU FP16 cross-entropy 溢出与 logits 非有限需分开解释。
"""

import gc
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from squid.checkpoints.qwen import load_qwen2_5
from squid.data.memmap_dataset import create_memmap_loader


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = (
    ROOT
    / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
)
HELDOUT_PATH = (
    ROOT
    / "data/token_store/blog_v1/heldout.uint32.bin"
)
REPORT_PATH = ROOT / "data/reports/i3-fp16-device-probe.json"

SCENARIOS = [
    ("mps", torch.float32),
    ("mps", torch.float16),
    ("mps", torch.bfloat16),
    ("cpu", torch.float32),
    ("cpu", torch.float16),
    ("cpu", torch.bfloat16),
]


def evaluate(device_name, dtype, input_ids, target_ids):
    device = torch.device(device_name)
    model = load_qwen2_5(
        WEIGHTS_PATH,
        cfg={"context_length": 128},
        device=device,
    )
    model = model.to(dtype=dtype)

    inputs = input_ids.to(device)
    targets = target_ids.to(device)

    with torch.no_grad():
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten(),
        )

    loss_value = loss.item()
    result = {
        "device": device_name,
        "dtype": str(dtype),
        "cross_entropy_loss": loss_value,
        "perplexity": math.exp(loss_value),
        "logits_all_finite": torch.isfinite(logits).all().item(),
        "logits_absmax": logits.abs().max().item(),
        "logits_mean_abs": logits.abs().mean().item(),
    }

    del model
    del logits
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return result


def main():
    loader = create_memmap_loader(
        HELDOUT_PATH,
        max_length=128,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        stride=128,
    )
    input_ids, target_ids = next(iter(loader))

    results = []
    for device_name, dtype in SCENARIOS:
        print(f"正在测量：device={device_name}, dtype={dtype}")
        try:
            result = evaluate(
                device_name,
                dtype,
                input_ids,
                target_ids,
            )
        except Exception as error:
            result = {
                "device": device_name,
                "dtype": str(dtype),
                "error": f"{type(error).__name__}: {error}",
            }

        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    report = {
        "experiment": "i3-fp16-device-probe",
        "evaluated_tokens": 128,
        "note": "单批诊断结果只用于本探针内部对照，不与十批 PPL 数字混用",
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n设备 / dtype 诊断报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()
