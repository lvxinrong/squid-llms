import json
import math
import time
from itertools import islice
from pathlib import Path

import torch

from squid.data.memmap_dataset import create_memmap_loader
from squid.device import get_device, memory_snapshot, reset_peak_memory_stats
from squid.model.v3_modern.gpt import GPTModelV3
from squid.train.common import calc_loss_batch


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = ROOT / "data/token_store/blog_v1/train.uint32.bin"
REPORT_PATH = ROOT / "data/reports/i2-compile-fp32-baseline.json"

MODEL_CONFIG = {
    "vocab_size": 151936,
    "emb_dim": 64,
    "ffn_dim": 256,
    "n_heads": 4,
    "n_kv_heads": 2,
    "n_layers": 2,
    "rope_theta": 1000000.0,
    "eps": 1e-6,
    "qkv_bias": True,
    "ffn_bias": False,
    "tie_word_embeddings": False,
    "drop_rate": 0.0,
    "context_length": 128,
    "attention_impl": "eager",
}
"""
CPU 与 GPU/MPS 设备之间的计算进度：让 CPU 等设备把已经提交的计算真正做完。
在 GPU/MPS 上，很多计算是异步提交的。可以理解为:
CPU：提交矩阵乘法任务
CPU：继续执行后面的 Python 代码
GPU/MPS：在设备端真正执行矩阵乘法
torch.mps.synchronize() 会让 CPU 等待 MPS 完成相关计算，测到的时间才包含真实设备执行时间。
synchronize() 就是让 CPU 等 GPU/MPS 把已提交的计算做完, 它同步的是计算进度 类似java中的 Future.get() 或者 Thread.join()
"""
def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def main():
    torch.manual_seed(123)
    device = get_device()

    train_loader = create_memmap_loader(
        TOKEN_PATH,
        max_length=128,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )

    model = GPTModelV3(MODEL_CONFIG).to(device)
    model = torch.compile(model)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    warmup_steps = 2
    measured_steps = 10
    batch_size = 2
    context_length = 128

    losses = []
    start = None

    for step, (inputs, targets) in enumerate(
        # islice()：只取前 N 个元素
        # islice(train_loader, 12)  表示只从 DataLoader 中取出前 12 个 batch，而不是遍历整个训练集，也不是取 12 条单独样本。
        # enumerate(..., 1)：从 1 开始编号
        # 表示一共取 12 个 batch：前 2 个用于预热，后 10 个用于正式测量。
        islice(train_loader, warmup_steps + measured_steps),
        1,
    ):
        # set_to_none=True 控制的是：清空梯度时，是把梯度写成全 0 张量，还是直接把 p.grad 设置为 None。
        # 它清空的是参数的 .grad，不是模型权重。
        # 这两者一个是“没有梯度对象”，一个是“存在一个值全为 0 的梯度张量”。
        # zero_grad(set_to_none=True) 是把上一轮梯度引用直接设为 None，而不是额外创建或保留一块全零梯度张量。它清理的是梯度，不是模型参数。
        optimizer.zero_grad(set_to_none=True)
        loss = calc_loss_batch(
            inputs,
            targets,
            model,
            device,
        )
        loss.backward()
        optimizer.step()

        # 先保留 Tensor，不在每个训练步骤里调用 item() 做设备同步。
        losses.append(loss.detach())

        if step == warmup_steps:
            synchronize(device)
            reset_peak_memory_stats(device)
            warmup_memory = memory_snapshot(device)
            start = time.perf_counter()

    synchronize(device)
    measured_memory = memory_snapshot(device)
    elapsed = time.perf_counter() - start

    loss_values = [loss.item() for loss in losses]
    assert all(math.isfinite(value) for value in loss_values)

    measured_tokens = measured_steps * batch_size * context_length
    tokens_per_second = measured_tokens / elapsed

    report = {
        "mode": "compile_fp32",
        "device": str(device),
        "dtype": "torch.float32",
        "model_config": MODEL_CONFIG,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "batch_size": batch_size,
        "context_length": context_length,
        "measured_tokens": measured_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
        "losses": loss_values,
        "attention_impl": "eager",
        "compiled": True,
        "memory_after_warmup": warmup_memory,
        "memory_after_measured": measured_memory,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nCompile + FP32 基线报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()