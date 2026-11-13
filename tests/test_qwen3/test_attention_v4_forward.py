import torch

from squid.model.v4_qwen3.layers import Qwen3Attention


def test_v4_attention_prefill_matches_incremental_decode():
    torch.manual_seed(123)
    attention = Qwen3Attention({
        "emb_dim": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "head_dim": 16,
        "context_length": 32,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_theta": 10000.0,
        "eps": 1e-6,
    }).eval()

    x = torch.randn(1, 4, 32)

    with torch.no_grad():
        expected, expected_past = attention(x)

        first, past_kv = attention(x[:, :2])
        second, past_kv = attention(
            x[:, 2:],
            past_kv=past_kv,
        )
        actual = torch.cat([first, second], dim=1)

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    # Cache 保存未扩展的 KV heads，而不是扩展后的 Q head 副本。
    assert past_kv[0].shape == (1, 2, 4, 16)
    assert past_kv[1].shape == (1, 2, 4, 16)

    for actual_tensor, expected_tensor in zip(
        past_kv,
        expected_past,
    ):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=1e-4,
            atol=1e-4,
        )
