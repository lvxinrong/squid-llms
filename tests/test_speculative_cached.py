import torch

from squid.generate.engine import generate_no_cache
from squid.generate.speculative import (
    generate_speculative_cached,
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


def make_models(same_weights):
    torch.manual_seed(123)
    target = GPTModelV3(TEST_CFG).eval()

    if same_weights:
        draft = GPTModelV3(TEST_CFG).eval()
        draft.load_state_dict(target.state_dict())
    else:
        torch.manual_seed(456)
        draft = GPTModelV3(TEST_CFG).eval()

    return draft, target


def check_matches_target(same_weights):
    draft, target = make_models(same_weights)
    idx = torch.tensor([[1, 2, 3]])

    expected = generate_no_cache(
        target,
        idx.clone(),
        max_new_tokens=8,
        context_size=32,
        temperature=0.0,
        eos_id=None,
    )

    actual = generate_speculative_cached(
        draft,
        target,
        idx.clone(),
        max_new_tokens=8,
        context_size=32,
        draft_lookahead=4,
    )

    assert torch.equal(actual, expected)


def test_cached_speculative_all_accepted_with_bonus():
    check_matches_target(same_weights=True)


def test_cached_speculative_rejection_and_rollback():
    check_matches_target(same_weights=False)

def test_cached_speculative_records_nonzero_ttft():
    draft, target = make_models(same_weights=False)

    _, stats = generate_speculative_cached(
        draft,
        target,
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=2,
        context_size=32,
        draft_lookahead=2,
        return_stats=True,
    )

    assert stats["ttft_seconds"] is not None
    assert stats["ttft_seconds"] > 0
