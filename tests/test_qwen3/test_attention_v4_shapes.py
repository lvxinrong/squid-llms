from squid.model.v4_qwen3.layers import Qwen3Attention


def test_v4_attention_uses_explicit_head_dim_and_qk_norm():
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
    })

    assert attention.W_query.weight.shape == (64, 32)
    assert attention.W_key.weight.shape == (32, 32)
    assert attention.W_value.weight.shape == (32, 32)
    assert attention.out_proj.weight.shape == (32, 64)

    assert attention.q_norm.scale.shape == (16,)
    assert attention.k_norm.scale.shape == (16,)

    assert attention.inv_freq.shape == (8,)
    assert attention.num_queries_per_kv == 2

    assert attention.W_query.bias is None
    assert attention.out_proj.bias is None
