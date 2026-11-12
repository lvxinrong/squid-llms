import torch

from squid.model.v3_modern.layers import RMSNorm


def test_rmsnorm_fp16_uses_fp32_accumulation():
    norm = RMSNorm({
        "emb_dim": 4,
        "eps": 1e-6,
    }).half().eval()

    x = torch.full(
        (1, 1, 4),
        619.0,
        dtype=torch.float16,
    )

    with torch.no_grad():
        output = norm(x)

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.ones_like(x))
