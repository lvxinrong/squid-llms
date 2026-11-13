"""I3 补验：在相同 attention 实现下实测 MHA / GQA 的 KV Cache 字节数与设备增量。

用途：只改变 KV head 数，隔离 GQA 对缓存空间的影响，并补齐 I0 拆出的实际占用验收。
输入：Qwen2.5-0.5B 的 attention 形状（24 层、14 Q heads、head_dim=64）与 36 个输入 token。
输出：data/reports/i3-kv-mha-gqa-mps.json。
边界：这是形状与内存探针，不评估模型语义；MHA 使用 n_kv_heads=n_heads 的同一实现，不是另一套模型权重。
"""

import gc
import json
from pathlib import Path

import torch
from torch import nn

from squid.device import get_device, memory_snapshot
from squid.generate.benchmark import kv_cache_nbytes
from squid.model.v3_modern.layers import GroupedQueryAttention


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data/reports/i3-kv-mha-gqa-mps.json"


def run_probe(n_kv_heads):
    torch.manual_seed(123)
    device = get_device()
    config = {
        "emb_dim": 896,
        "n_heads": 14,
        "n_kv_heads": n_kv_heads,
        "context_length": 128,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "rope_theta": 1000000.0,
        "attention_impl": "eager",
    }
    n_layers = 24
    token_count = 36
    head_dim = config["emb_dim"] // config["n_heads"]

    attentions = nn.ModuleList([
        GroupedQueryAttention(config)
        for _ in range(n_layers)
    ]).to(device)
    x = torch.randn(
        1,
        token_count,
        config["emb_dim"],
        device=device,
    )
    memory_before = memory_snapshot(device)

    caches = []
    with torch.no_grad():
        for attention in attentions:
            context, present = attention(x)
            caches.append(present)
        del context

    memory_after = memory_snapshot(device)
    actual_bytes = kv_cache_nbytes(caches)
    expected_per_token = (
        2
        * n_layers
        * n_kv_heads
        * head_dim
        * torch.float32.itemsize
    )

    assert actual_bytes == expected_per_token * token_count

    result = {
        "attention_mode": "MHA" if n_kv_heads == config["n_heads"] else "GQA",
        "n_layers": n_layers,
        "n_heads": config["n_heads"],
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "tokens": token_count,
        "dtype": "torch.float32",
        "kv_cache_bytes": actual_bytes,
        "kv_cache_bytes_per_token": actual_bytes / token_count,
        "expected_bytes_per_token": expected_per_token,
        "current_allocated_delta_bytes": (
            memory_after["current_allocated_bytes"]
            - memory_before["current_allocated_bytes"]
        ),
        "driver_allocated_delta_bytes": (
            memory_after["driver_allocated_bytes"]
            - memory_before["driver_allocated_bytes"]
            if device.type == "mps"
            else None
        ),
        "memory_before": memory_before,
        "memory_after": memory_after,
    }

    del caches
    del attentions
    del x
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return result


def main():
    mha = run_probe(n_kv_heads=14)
    gqa = run_probe(n_kv_heads=2)

    assert gqa["kv_cache_bytes"] * 7 == mha["kv_cache_bytes"]

    report = {
        "experiment": "i3-kv-mha-gqa-probe",
        "device": str(get_device()),
        "shape_source": "Qwen2.5-0.5B attention only",
        "mha": mha,
        "gqa": gqa,
        "gqa_over_mha_bytes": (
            gqa["kv_cache_bytes"] / mha["kv_cache_bytes"]
        ),
        "gqa_reduction_percent": (
            1 - gqa["kv_cache_bytes"] / mha["kv_cache_bytes"]
        ) * 100,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nMHA / GQA KV 报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()
