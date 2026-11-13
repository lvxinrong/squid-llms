import torch

from squid.generate.engine import generate, generate_stream
from squid.model.v3_modern.gpt import GPTModelV3


TEST_CFG = {
    "vocab_size": 128,
    "emb_dim": 16,
    "ffn_dim": 32,
    "n_heads": 4,
    "n_kv_heads": 2,
    "n_layers": 2,
    "rope_theta": 10000.0,
    "eps": 1e-6,
    "qkv_bias": True,
    "ffn_bias": False,
    "tie_word_embeddings": False,
    "drop_rate": 0.0,
    "context_length": 32,
}


def test_generate_stream_matches_generate_and_returns_cache():
    torch.manual_seed(123)
    model = GPTModelV3(TEST_CFG).eval()
    idx = torch.tensor([[1, 2, 3, 4]])

    expected, expected_past = generate(
        model,
        idx.clone(),
        max_new_tokens=5,
        context_size=32,
        return_past_kv=True,
    )

    stream = generate_stream(
        model,
        idx.clone(),
        max_new_tokens=5,
        context_size=32,
    )
    emitted = []

    while True:
        try:
            next_id = next(stream)
            emitted.append(next_id.item())
        except StopIteration as stop:
            actual, actual_past = stop.value
            break

    assert torch.equal(actual, expected)
    assert emitted == expected[0, idx.shape[1]:].tolist()
    assert len(actual_past) == len(expected_past)

    for actual_layer, expected_layer in zip(actual_past, expected_past):
        for actual_tensor, expected_tensor in zip(actual_layer, expected_layer):
            assert torch.equal(actual_tensor, expected_tensor)
