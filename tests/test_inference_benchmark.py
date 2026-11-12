import torch

from squid.generate.benchmark import (
    kv_cache_nbytes,
    measure_fixed_length_generation,
)
from squid.generate.engine import generate, generate_no_cache
from squid.model.v3_modern.gpt import GPTModelV3


TEST_CFG = {
    "vocab_size": 64,
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


def test_cached_benchmark_matches_existing_generate():
    torch.manual_seed(123)
    model = GPTModelV3(TEST_CFG).eval()
    idx = torch.tensor([[1, 2, 3, 4]])

    expected = generate(
        model,
        idx.clone(),
        max_new_tokens=5,
        context_size=32,
    )
    actual, metrics = measure_fixed_length_generation(
        model,
        idx.clone(),
        max_new_tokens=5,
        context_size=32,
        use_cache=True,
    )

    assert torch.equal(actual, expected)
    assert metrics["generated_tokens"] == 5
    assert metrics["decode_tokens"] == 4
    assert metrics["ttft_seconds"] > 0
    assert metrics["decode_seconds"] > 0


def test_no_cache_benchmark_matches_existing_generate():
    torch.manual_seed(123)
    model = GPTModelV3(TEST_CFG).eval()
    idx = torch.tensor([[1, 2, 3, 4]])

    expected = generate_no_cache(
        model,
        idx.clone(),
        max_new_tokens=5,
        context_size=32,
    )
    actual, metrics = measure_fixed_length_generation(
        model,
        idx.clone(),
        max_new_tokens=5,
        context_size=32,
        use_cache=False,
    )

    assert torch.equal(actual, expected)
    assert metrics["kv_cache_after_prefill_bytes"] == 0


def test_kv_cache_nbytes_uses_real_tensor_storage():
    model = GPTModelV3(TEST_CFG).eval()
    idx = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        _, past_kv = model(idx, use_cache=True)

    head_dim = TEST_CFG["emb_dim"] // TEST_CFG["n_heads"]
    expected = (
        2
        * TEST_CFG["n_layers"]
        * idx.shape[0]
        * idx.shape[1]
        * TEST_CFG["n_kv_heads"]
        * head_dim
        * torch.float32.itemsize
    )
    assert kv_cache_nbytes(past_kv) == expected
