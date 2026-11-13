import torch

from squid.model.v4_qwen3.gpt import GPTModelV4


def test_v4_model_forward_cache_and_weight_tying():
    model = GPTModelV4({
        "vocab_size": 128,
        "emb_dim": 32,
        "ffn_dim": 64,
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_layers": 2,
        "head_dim": 16,
        "rope_theta": 10000.0,
        "eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "drop_rate": 0.0,
        "context_length": 32,
        "ffn_bias": False,
        "tie_word_embeddings": True,
    }).eval()

    assert model.out_head.weight is model.tok_emb.weight

    idx = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        logits, past_kv = model(idx, use_cache=True)

    assert logits.shape == (1, 4, 128)
    assert torch.isfinite(logits).all()
    assert len(past_kv) == 2
    assert past_kv[0][0].shape == (1, 2, 4, 16)
