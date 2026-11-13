import torch

from squid.model.v4_qwen3.layers import TransformerBlock


def test_v4_transformer_block_forward_and_cache():
    torch.manual_seed(123)
    block = TransformerBlock({
        "emb_dim": 32,
        "ffn_dim": 64,
        "n_heads": 4,
        "n_kv_heads": 2,
        "head_dim": 16,
        "context_length": 32,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_theta": 10000.0,
        "eps": 1e-6,
        "ffn_bias": False,
        "drop_rate": 0.0,
    }).eval()

    x = torch.randn(1, 4, 32)

    with torch.no_grad():
        output, past_kv = block(x)

    assert output.shape == (1, 4, 32)
    assert torch.isfinite(output).all()

    assert len(past_kv) == 2
    assert past_kv[0].shape == (1, 2, 4, 16)
    assert past_kv[1].shape == (1, 2, 4, 16)
