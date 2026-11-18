import runpy
from pathlib import Path

from squid.eval.pairwise import reconcile_position_orders


ROOT = Path(__file__).resolve().parents[1]
length_namespace = runpy.run_path(str(
    ROOT / "scripts/i5_length_control.py"
))
judge_namespace = runpy.run_path(str(
    ROOT / "scripts/i5_length_judge.py"
))

build_pairs = length_namespace["build_pairs"]
annotate_result = judge_namespace["annotate_result"]
build_summary = judge_namespace["build_summary"]


def result_for(pair, ab_winner, ba_winner):
    return {
        "pair_id": pair["pair_id"],
        "orders": {
            "AB": {"winner": ab_winner},
            "BA": {"winner": ba_winner},
        },
        "comparison": reconcile_position_orders(
            ab_winner,
            ba_winner,
        ),
    }


def test_padding_winner_is_longer_but_disagrees_with_expectation():
    pair = build_pairs()[1]

    result = annotate_result(
        result_for(pair, "B", "A"),
        pair,
    )

    assert result["comparison"]["consistent"] is True
    assert result["comparison"]["consensus_winner"] == "B"
    assert result["consensus_role"] == "padded"
    assert result["length_choice"] == "longer"
    assert result["expected_agreement"] is False


def test_informative_winner_is_not_scored_as_length_bias():
    pair = build_pairs()[2]

    result = annotate_result(
        result_for(pair, "B", "A"),
        pair,
    )

    assert result["consensus_role"] == "informative"
    assert result["length_choice"] == "longer"
    assert result["expected_winner"] is None
    assert result["expected_agreement"] is None


def test_unresolved_expectation_is_not_counted_as_disagreement():
    pair = build_pairs()[1]
    result = annotate_result(
        result_for(pair, "A", "A"),
        pair,
    )

    summary = build_summary([result])

    assert result["comparison"]["consensus_winner"] is None
    assert result["length_choice"] == "unresolved"
    assert summary["expected_comparable"] == 0
    assert summary["expected_disagreement"] == 0
    assert summary["expected_unresolved"] == 1
