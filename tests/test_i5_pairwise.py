import pytest

from squid.eval.pairwise import (
    build_pairwise_prompt,
    canonicalize_winner,
    judge_pair_both_orders,
    parse_pairwise_winner,
    reconcile_position_orders, build_pairwise_prompt_v2,
)


@pytest.mark.parametrize(
    "ab_winner,ba_winner,consensus,consistent,decisive_flip",
    [
        ("A", "B", "A", True, False),
        ("B", "A", "B", True, False),
        ("A", "A", None, False, True),
        ("B", "B", None, False, True),
        ("tie", "tie", "tie", True, False),
        ("A", "tie", None, False, False),
        ("tie", "A", None, False, False),
        ("B", "tie", None, False, False),
    ],
)
def test_reconcile_position_orders(
    ab_winner,
    ba_winner,
    consensus,
    consistent,
    decisive_flip,
):
    result = reconcile_position_orders(ab_winner, ba_winner)

    assert result["consensus_winner"] == consensus
    assert result["consistent"] is consistent
    assert result["position_sensitive"] is not consistent
    assert result["decisive_flip"] is decisive_flip


def test_reconcile_restores_original_coordinates():
    result = reconcile_position_orders("A", "A")

    assert result["ab_canonical_winner"] == "A"
    assert result["ba_canonical_winner"] == "B"


def test_invalid_winner_is_rejected():
    with pytest.raises(ValueError, match="非法 winner"):
        canonicalize_winner("C")


def make_pair():
    return {
        "pair_id": "test-pair",
        "history": [],
        "user": "119 + 72 等于多少？",
        "expected_answer": "191",
        "responses": {
            "A": "正确",
            "B": "错误",
        },
        "human": {
            "winner": "A",
            "notes": "人工标签不能进入评委 prompt",
        },
    }


def test_prompt_swaps_responses_and_hides_human_labels():
    pair = make_pair()

    ab_prompt = build_pairwise_prompt(pair)
    ba_prompt = build_pairwise_prompt(pair, swapped=True)

    assert "[回答 A]\n正确" in ab_prompt
    assert "[回答 B]\n错误" in ab_prompt

    assert "[回答 A]\n错误" in ba_prompt
    assert "[回答 B]\n正确" in ba_prompt

    assert "人工标签不能进入评委 prompt" not in ab_prompt
    assert "人工标签不能进入评委 prompt" not in ba_prompt


def test_content_aware_fake_judge_is_consistent():
    calls = []

    def choose_correct(prompt, model):
        calls.append(prompt)
        winner = (
            "A"
            if "[回答 A]\n正确" in prompt
            else "B"
        )
        return {
            "content": winner,
            "model": model,
        }

    result = judge_pair_both_orders(
        make_pair(),
        choose_correct,
        model="fake-content-judge",
    )

    assert len(calls) == 2
    assert result["orders"]["AB"]["winner"] == "A"
    assert result["orders"]["BA"]["winner"] == "B"
    assert result["comparison"]["consistent"] is True
    assert result["comparison"]["consensus_winner"] == "A"
    assert result["comparison"]["decisive_flip"] is False


def test_always_a_fake_judge_produces_decisive_flip():
    def always_choose_a(prompt, model):
        return {
            "content": "A",
            "model": model,
        }

    result = judge_pair_both_orders(
        make_pair(),
        always_choose_a,
        model="fake-position-judge",
    )

    assert result["orders"]["AB"]["winner"] == "A"
    assert result["orders"]["BA"]["winner"] == "A"
    assert result["comparison"]["consistent"] is False
    assert result["comparison"]["consensus_winner"] is None
    assert result["comparison"]["decisive_flip"] is True


def test_pairwise_parser_rejects_extra_explanation():
    assert parse_pairwise_winner(" A\n") == "A"
    assert parse_pairwise_winner("tie") == "tie"

    with pytest.raises(ValueError, match="无法解析"):
        parse_pairwise_winner("A，因为回答更正确")


def test_existing_order_reuses_saved_result():
    calls = []
    after_order_calls = []
    existing = {
        "orders": {
            "AB": {
                "swapped": False,
                "winner": "A",
                "judge": {
                    "content": "A",
                    "model": "fake-content-judge",
                },
            }
        }
    }

    def choose_correct(prompt, model):
        calls.append(prompt)
        winner = (
            "A"
            if "[回答 A]\n正确" in prompt
            else "B"
        )
        return {
            "content": winner,
            "model": model,
        }

    result = judge_pair_both_orders(
        make_pair(),
        choose_correct,
        model="fake-content-judge",
        existing=existing,
        after_order=lambda orders: after_order_calls.append(
            dict(orders)
        ),
    )

    assert len(calls) == 1
    assert len(after_order_calls) == 1
    assert result["orders"]["AB"] == existing["orders"]["AB"]
    assert result["orders"]["BA"]["winner"] == "B"
    assert result["comparison"]["consistent"] is True
    assert result["comparison"]["consensus_winner"] == "A"

def test_v2_prompt_makes_independent_verification_explicit():
    pair = make_pair()

    v1_prompt = build_pairwise_prompt(pair)
    v2_prompt = build_pairwise_prompt_v2(pair)

    assert "独立验算或推理可以构成实质优势" in v2_prompt
    assert "额外验算不是强制要求" in v2_prompt
    assert "重复同一种计算" in v2_prompt
    assert "独立验算或推理可以构成实质优势" not in v1_prompt
    assert "人工标签不能进入评委 prompt" not in v2_prompt


def test_v2_prompt_builder_can_be_injected():
    prompts = []

    def choose_correct(prompt, model):
        prompts.append(prompt)
        winner = (
            "A"
            if "[回答 A]\n正确" in prompt
            else "B"
        )
        return {
            "content": winner,
            "model": model,
        }

    result = judge_pair_both_orders(
        make_pair(),
        choose_correct,
        model="fake-v2-judge",
        prompt_builder=build_pairwise_prompt_v2,
    )

    assert len(prompts) == 2
    assert all(
        "独立验算或推理可以构成实质优势" in prompt
        for prompt in prompts
    )
    assert result["comparison"]["consistent"] is True
    assert result["comparison"]["consensus_winner"] == "A"


