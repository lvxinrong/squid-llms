import torch

from squid.model.v2_rope.gpt import GPTModelRoPE
from squid.model.v2_rope.rope import (
    compute_inv_freq,
    compute_cos_sin,
    apply_rope,
)

def test_rope_zero_position_and_norm():
    torch.manual_seed(42)

    # B=2，H=3，T=4，head_dim=8
    x = torch.randn(2, 3, 4, 8)
    position_ids = torch.arange(4)

    inv_freq = compute_inv_freq(head_dim=8)
    cos, sin = compute_cos_sin(position_ids, inv_freq)
    out = apply_rope(x, cos, sin)

    # 旋转不改变形状
    assert out.shape == x.shape

    # 位置 0：cos=1、sin=0，应该完全不变
    torch.testing.assert_close(
        out[:, :, 0, :],
        x[:, :, 0, :],
    )

    # 每个位置的完整 Q/K 向量长度保持不变
    torch.testing.assert_close(
        out.norm(dim=-1),
        x.norm(dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )


def rope_dot(q, k, q_pos, k_pos, inv_freq):
    q_cos, q_sin = compute_cos_sin(torch.tensor([q_pos]), inv_freq)
    k_cos, k_sin = compute_cos_sin(torch.tensor([k_pos]), inv_freq)

    rotated_q = apply_rope(q, q_cos, q_sin)
    rotated_k = apply_rope(k, k_cos, k_sin)

    # 对应注意力里的完整 Q · K
    return (rotated_q * rotated_k).sum()


def test_rope_relative_position():
    torch.manual_seed(42)

    # 同一个 Q/K，只改变它们所在的位置
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    inv_freq = compute_inv_freq(head_dim=8)

    original = rope_dot(q, k, 3, 6, inv_freq)
    shifted = rope_dot(q, k, 103, 106, inv_freq)
    different_distance = rope_dot(q, k, 3, 7, inv_freq)

    # 相对距离都是 3，点积应该一致
    torch.testing.assert_close(
        original,
        shifted,
        atol=1e-5,
        rtol=1e-5,
    )

    # 相对距离变成 4，点积应该发生变化；同时防止 RoPE 写成恒等函数也通过
    assert not torch.isclose(
        original,
        different_distance,
        atol=1e-5,
        rtol=1e-5,
    )


def test_gpt_rope_cache_equivalence():
    torch.manual_seed(123)

    cfg = {
        "vocab_size": 50,
        "context_length": 8,
        "emb_dim": 16,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModelRoPE(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (2, 5))

    full_logits = model(idx)

    prefill_logits, past = model(idx[:, :3], use_cache=True)
    parts = [prefill_logits]
    for i in range(3, idx.shape[1]):
        logits, past = model(
            idx[:, i:i + 1],
            past_kv=past,
            use_cache=True,
        )
        parts.append(logits)

    cache_logits = torch.cat(parts, dim=1)

    torch.testing.assert_close(
        full_logits,
        cache_logits,
        atol=1e-5,
        rtol=1e-5,
    )