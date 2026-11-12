"""I3 FP16 排障：逐层 hook RMSNorm，检查输入幅度、平方均值和输出是否有限。

用途：直接验证 `x.pow(2).mean()` 是否在 FP16 下溢出，并定位首个异常 Transformer 层。
输入：真实 Qwen 与同一批 128 个 heldout tokens，对照 MPS 上 FP32 / FP16 / BF16。
输出：data/reports/i3-fp16-rmsnorm-probe.json。
边界：输出有限不代表中间值正确，需结合 mean_square 判断。
"""

import gc
import json
from pathlib import Path

import torch

from squid.checkpoints.qwen import load_qwen2_5
from squid.data.memmap_dataset import create_memmap_loader
from squid.model.v3_modern.layers import RMSNorm


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = (
    ROOT
    / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
)
HELDOUT_PATH = (
    ROOT
    / "data/token_store/blog_v1/heldout.uint32.bin"
)
REPORT_PATH = ROOT / "data/reports/i3-fp16-rmsnorm-probe.json"


def probe(dtype, input_ids):
    device = torch.device("mps")
    model = load_qwen2_5(
        WEIGHTS_PATH,
        cfg={"context_length": 128},
        device=device,
    )
    model = model.to(dtype=dtype)

    records = []

    def make_hook(name):
        def hook(module, args, output):
            x = args[0].detach()
            mean_square = x.pow(2).mean(
                dim=-1,
                keepdim=True,
            )

            records.append({
                "name": name,
                "input_absmax": x.abs().max().item(),
                "mean_square_max": mean_square.max().item(),
                "mean_square_all_finite": (
                    torch.isfinite(mean_square).all().item()
                ),
                "output_all_finite": (
                    torch.isfinite(output).all().item()
                ),
            })

        return hook

    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            module.register_forward_hook(make_hook(name))

    with torch.no_grad():
        model(input_ids.to(device))

    result = {
        "dtype": str(dtype),
        "records": records,
        "nonfinite_mean_square_layers": [
            record["name"]
            for record in records
            if not record["mean_square_all_finite"]
        ],
        "nonfinite_output_layers": [
            record["name"]
            for record in records
            if not record["output_all_finite"]
        ],
    }

    del model
    gc.collect()
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
    input_ids, _ = next(iter(loader))

    results = []
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        print(f"正在检查 RMSNorm：{dtype}")
        result = probe(dtype, input_ids)
        results.append(result)
        print(json.dumps({
            "dtype": result["dtype"],
            "nonfinite_mean_square_layers": (
                result["nonfinite_mean_square_layers"]
            ),
            "nonfinite_output_layers": (
                result["nonfinite_output_layers"]
            ),
        }, ensure_ascii=False))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nRMSNorm 诊断报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()
