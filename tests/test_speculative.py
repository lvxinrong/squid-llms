import pytest
import torch

from squid.generate.speculative import (
    truncate_past_kv,
    verify_greedy_draft,
)

def test_all_draft_tokens_accepted_with_bonus():
    result = verify_greedy_draft(
        draft_ids=[10, 11, 12],
        target_ids=[10, 11, 12, 99],
    )

    assert result == {
        "output_ids": [10, 11, 12, 99],
        "accepted_draft_count": 3,
        "rejected_draft_count": 0,
        "used_target_correction": False,
    }


def test_rejection_uses_target_correction_and_rolls_back_suffix():
    result = verify_greedy_draft(
        draft_ids=[10, 11, 12, 13],
        target_ids=[10, 11, 99, 98, 97],
    )

    assert result["output_ids"] == [10, 11, 99]
    assert result["accepted_draft_count"] == 2
    assert result["rejected_draft_count"] == 2
    assert result["used_target_correction"] is True


def test_first_draft_token_rejected():
    result = verify_greedy_draft(
        draft_ids=[10, 11],
        target_ids=[99, 98, 97],
    )



    assert result["output_ids"] == [99]
    assert result["accepted_draft_count"] == 0
    assert result["rejected_draft_count"] == 2


def test_target_prediction_length_must_match():
    with pytest.raises(ValueError, match="多一个"):
        verify_greedy_draft(
            draft_ids=[10, 11],
            target_ids=[10, 11],
        )

def test_last_draft_token_rejected():
    result = verify_greedy_draft(
        draft_ids=[10, 11, 12],
        target_ids=[10, 11, 99, 100],
    )

    assert result["output_ids"] == [10, 11, 99]
    assert result["accepted_draft_count"] == 2
    assert result["rejected_draft_count"] == 1
    assert result["used_target_correction"] is True


def test_truncate_past_kv_keeps_only_accepted_prefix():
    key = torch.arange(
        2 * 6 * 4
    ).reshape(1, 2, 6, 4)
    value = key + 100
    past_kv = [
        (key, value),
        (key + 1000, value + 1000),
    ]

    trimmed = truncate_past_kv(past_kv, length=3)

    assert len(trimmed) == 2
    assert trimmed[0][0].shape == (1, 2, 3, 4)

    assert torch.equal(
        trimmed[0][0],
        key[:, :, :3],
    )
    assert torch.equal(
        trimmed[0][1],
        value[:, :, :3],
    )
    assert torch.equal(
        trimmed[1][0],
        (key + 1000)[:, :, :3],
    )

    trimmed[0][0][0, 0, 0, 0] = 999
    assert key[0, 0, 0, 0].item() == 0


def test_truncate_past_kv_rejects_invalid_length():
    key = torch.zeros(1, 2, 6, 4)
    past_kv = [(key, key.clone())]

    with pytest.raises(ValueError, match="当前长度"):
        truncate_past_kv(past_kv, length=7)


