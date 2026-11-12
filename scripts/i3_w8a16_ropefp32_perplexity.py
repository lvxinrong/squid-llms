"""I3 FP16 假设对照：模型使用 FP16，但将每层 RoPE inv_freq 恢复为 FP32 后复测 PPL。

用途：检验 inv_freq 半精度舍入是否能解释 FP16 PPL 异常；这是排查变量，不是主线量化配置。
输入：真实 Qwen 与 blog_v1 heldout 的 5120 tokens；输出 data/reports/i3-w8a16-ropefp32-perplexity-mps.json。
边界：当时结果与原始 FP16 完全一致，说明仅恢复 inv_freq 不足以修复；最终根因是 RMSNorm FP16 平方溢出。
"""

import json
import math
from pathlib import Path
from squid.checkpoints.qwen import load_qwen2_5
from squid.data.memmap_dataset import create_memmap_loader
from squid.device import get_device, memory_snapshot
from squid.quantize.convert import quantize_model_int8
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
REPORT_PATH = (
    ROOT
    / "data/reports/i3-w8a16-ropefp32-perplexity-mps.json"
)

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
    # model.half() 会将浮点参数和浮点 buffer 转成 FP16；INT8、INT4、uint8、bool 等整数或布尔张量不会被转成浮点。
    model = model.half()

    for block in model.trf_block:
        # inv_freq 保持 FP32；
        block.att.inv_freq = block.att.inv_freq.float()

    reference = evaluate(model, loader, device)

    replaced = quantize_model_int8(model)
    assert len(replaced) == 168

    quantized = evaluate(model, loader, device)

    reference_ppl = reference["perplexity"]
    quantized_ppl = quantized["perplexity"]

    report = {
        "experiment": "i3-w8a16-ropefp32-perplexity",
        "model": "Qwen2.5-0.5B-Instruct",
        "device": str(device),
        "data": "blog_v1 heldout",
        "context_size": context_size,
        "batch_size": batch_size,
        "evaluated_batches": len(loader),
        "evaluated_tokens": (
            len(loader) * batch_size * context_size
        ),
        "weight_path": "per-row INT8, out_head excluded",
        "compute_dtype": "torch.float16",
        "reference_fp16": reference,
        "quantized_int8": quantized,
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
