import time

import torch

from squid.generate.engine import sample_next


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def kv_cache_nbytes(past_kv):
    if past_kv is None:
        return 0
    return sum(
        tensor.nelement() * tensor.element_size()
        for layer_kv in past_kv
        for tensor in layer_kv
    )


def measure_fixed_length_generation(
    model,
    idx,
    max_new_tokens,
    context_size,
    use_cache=True,
    temperature=0.0,
    top_k=None,
):
    """测量单请求、固定长度的生成，不启用 EOS 提前停止。"""
    if idx.ndim != 2 or idx.shape[0] != 1:
        raise ValueError("推理基线只测量 batch_size=1 的单个请求")
    if idx.shape[1] < 1:
        raise ValueError("prompt 至少需要一个 token")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须大于等于 1")

    device = idx.device
    prefill_kv_bytes = 0
    past_kv = None

    with torch.no_grad():
        synchronize(device)
        started_at = time.perf_counter()

        if use_cache:
            idx_cond = idx[:, -context_size:]
            logits, past_kv = model(idx_cond, use_cache=True)
            prefill_kv_bytes = kv_cache_nbytes(past_kv)

        first_token_at = None
        last_token_at = None

        for step in range(max_new_tokens):
            if not use_cache:
                idx_cond = idx[:, -context_size:]
                logits = model(idx_cond, use_cache=False)

            idx_next = sample_next(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
            )
            idx = torch.cat([idx, idx_next], dim=1)

            if step == 0:
                synchronize(device)
                first_token_at = time.perf_counter()

            if step == max_new_tokens - 1:
                synchronize(device)
                last_token_at = time.perf_counter()

            if use_cache:
                logits, past_kv = model(
                    idx_next,
                    past_kv=past_kv,
                    use_cache=True,
                )

        synchronize(device)
        finished_at = time.perf_counter()

    decode_tokens = max_new_tokens - 1
    decode_seconds = last_token_at - first_token_at

    return idx, {
        "use_cache": use_cache,
        "generated_tokens": max_new_tokens,
        "decode_tokens": decode_tokens,
        "ttft_seconds": first_token_at - started_at,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (
            decode_tokens / decode_seconds if decode_tokens > 0 else None
        ),
        "total_seconds": finished_at - started_at,
        "kv_cache_after_prefill_bytes": prefill_kv_bytes,
    }
