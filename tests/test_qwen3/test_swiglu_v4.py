import torch
from torch.nn import functional as F

from squid.model.v4_qwen3.layers import SwiGLUFFN


def test_v4_swiglu_matches_manual_formula():
    model = SwiGLUFFN({
        "emb_dim": 2,
        "ffn_dim": 2,
        "ffn_bias": False,
    }).eval()

    with torch.no_grad():
        for layer in (
            model.gate_proj,
            model.up_proj,
            model.down_proj,
        ):
            layer.weight.copy_(torch.eye(2))

    x = torch.tensor([
        [1.0, 2.0],
        [-0.5, 0.3],
    ])

    actual = model(x)
    expected = F.silu(x) * x

    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
