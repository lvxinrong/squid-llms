import torch

from squid.generate.engine import generate_no_cache
from squid.generate.speculative import (
    generate_speculative_no_cache,
)
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


def test_speculative_matches_target_when_models_are_identical():
    torch.manual_seed(123)
    target = GPTModelV3(TEST_CFG).eval()
    draft = GPTModelV3(TEST_CFG).eval()
    draft.load_state_dict(target.state_dict())

    idx = torch.tensor([[1, 2, 3]])
    expected = generate_no_cache(
        target,
        idx.clone(),
        max_new_tokens=6,
        context_size=32,
        temperature=0.0,
        eos_id=None,
    )

    actual = generate_speculative_no_cache(
        draft,
        target,
        idx.clone(),
        max_new_tokens=6,
        context_size=32,
        draft_lookahead=4,
    )

    assert torch.equal(actual, expected)


def test_speculative_matches_target_with_different_draft():
    torch.manual_seed(123)
    target = GPTModelV3(TEST_CFG).eval()

    torch.manual_seed(456)
    draft = GPTModelV3(TEST_CFG).eval()

    idx = torch.tensor([[1, 2, 3]])
    expected = generate_no_cache(
        target,
        idx.clone(),
        max_new_tokens=6,
        context_size=32,
        temperature=0.0,
        eos_id=None,
    )

    actual = generate_speculative_no_cache(
        draft,
        target,
        idx.clone(),
        max_new_tokens=6,
        context_size=32,
        draft_lookahead=4,
    )

    assert torch.equal(actual, expected)
