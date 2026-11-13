import torch

from squid.model.v4_qwen3.rope import (
    apply_rope,
    compute_cos_sin,
    compute_inv_freq,
)


def test_v4_rope_inverse_frequency():
    inv_freq = compute_inv_freq(
        head_dim=4,
        base=10000.0,
    )

    torch.testing.assert_close(
        inv_freq,
        torch.tensor([1.0, 0.01]),
    )


def test_v4_rope_position_zero_is_identity():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    inv_freq = compute_inv_freq(
        head_dim=4,
        base=10000.0,
    )
    cos, sin = compute_cos_sin(
        torch.tensor([0]),
        inv_freq,
    )

    actual = apply_rope(x, cos, sin)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, x)
