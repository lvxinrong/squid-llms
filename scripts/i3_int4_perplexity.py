"""I3-2 W4A32 困惑度对照：测量朴素逐行 packed INT4 对 heldout next-token 预测的影响。

用途：用与 FP32 / INT8 完全相同的 5120 个 heldout tokens 评估 INT4 的质量代价。
输入：真实 Qwen2.5-0.5B-Instruct 和 blog_v1 heldout，context=128、batch=4、10 批。
输出：data/reports/i3-int4-perplexity-mps.json。
边界：结果仅描述当前逐行量化实现，不推广为所有 INT4 方法的固有能力。
"""

import json
import math
from pathlib import Path
from squid.checkpoints.qwen import load_qwen2_5
from squid.data.memmap_dataset import create_memmap_loader
from squid.device import get_device, memory_snapshot
from squid.quantize.convert import quantize_model_int4
from squid.train.common import calc_loss_loader


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = (
    ROOT
    / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
)
HELDOUT_PATH = (
    ROOT
    / "data/token_store/blog_v1/heldout.uint32.bin"
)
REPORT_PATH = ROOT / "data/reports/i3-int4-perplexity-mps.json"


def evaluate(model, loader, device):
    model.eval()
    loss = calc_loss_loader(
        loader,
        model,
        device,
        num_batches=None,
    )

    return {
        "cross_entropy_loss": loss,
        # PPL perplexity 困惑度 可以粗略地把 PPL 理解为模型预测下一个 token 时的“有效选择数量”：PPL 越高，模型越像在更多候选之间犹豫。
        "perplexity": math.exp(loss),
        "memory_after_evaluation": memory_snapshot(device),
    }


def main():
    device = get_device()
    context_size = 128
    batch_size = 4

    loader = create_memmap_loader(
        HELDOUT_PATH,
        max_length=context_size,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        stride=context_size,
    )

    model = load_qwen2_5(
        WEIGHTS_PATH,
        cfg={"context_length": context_size},
        device=device,
    )

    reference = evaluate(model, loader, device)

    replaced = quantize_model_int4(model)
    assert len(replaced) == 168

    quantized = evaluate(model, loader, device)

    reference_ppl = reference["perplexity"]
    quantized_ppl = quantized["perplexity"]

    report = {
        "experiment": "i3-int4-perplexity",
        "model": "Qwen2.5-0.5B-Instruct",
        "device": str(device),
        "data": "blog_v1 heldout",
        "context_size": context_size,
        "batch_size": batch_size,
        "evaluated_batches": len(loader),
        "evaluated_tokens": (
            len(loader) * batch_size * context_size
        ),
        "weight_path": "per-row INT4 packed, out_head excluded",
        "compute_dtype": "torch.float32",
        "reference_fp32": reference,
        "quantized_int4": quantized,
        "perplexity_absolute_change": (
            quantized_ppl - reference_ppl
        ),
        "perplexity_change_percent": (
            (quantized_ppl - reference_ppl)
            / reference_ppl
            * 100
        ),
        "perplexity_ratio_quantized_over_fp32": (
            quantized_ppl / reference_ppl
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nPPL 对照报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()
