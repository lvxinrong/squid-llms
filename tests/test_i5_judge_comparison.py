import runpy
from pathlib import Path

import pytest

from squid.eval.pairwise import reconcile_position_orders


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i5_judge_comparison.py"
))
build_comparison = namespace["build_comparison"]


def make_blind():
    return {
        "status": "labeled",
        "pairs": [
            {
                "pair_id": pair_id,
                "human": {
                    "winner": winner,
                    "reason_codes": [],
                },
            }
            for pair_id, winner in (
                ("p1", "tie"),
                ("p2", "B"),
                ("p3", "A"),
            )
        ],
    }


def make_result(pair_id, human_winner, ab_winner, ba_winner):
    return {
        "pair_id": pair_id,
        "human_winner": human_winner,
        "orders": {
            "AB": {"winner": ab_winner},
            "BA": {"winner": ba_winner},
        },
        "comparison": reconcile_position_orders(
            ab_winner,
            ba_winner,
        ),
    }


def make_report(results, blind_sha256="test-hash"):
    return {
        "status": "complete",
        "prompt_version": "test-prompt",
        "blind_sha256": blind_sha256,
        "provider": "test-provider",
        "judge_model": "test-model",
        "query_settings": {},
        "pairs": results,
    }


def test_three_way_comparison_counts_agreement_and_unresolved():
    deepseek = make_report([
        make_result("p1", "tie", "tie", "A"),
        make_result("p2", "B", "tie", "tie"),
        make_result("p3", "A", "A", "B"),
    ])
    qwen = make_report([
        make_result("p1", "tie", "tie", "tie"),
        make_result("p2", "B", "tie", "tie"),
        make_result("p3", "A", "A", "B"),
    ])

    report = build_comparison(
        make_blind(),
        deepseek,
        qwen,
        "test-hash",
    )
    summary = report["summary"]

    assert summary["judges"]["deepseek"]["human_agreement"] == 1
    assert summary["judges"]["deepseek"]["human_disagreement"] == 1
    assert summary["judges"]["deepseek"]["human_unresolved"] == 1

    assert summary["judges"]["qwen"]["human_agreement"] == 2
    assert summary["judges"]["qwen"]["human_disagreement"] == 1
    assert summary["judges"]["qwen"]["human_unresolved"] == 0

    cross = summary["cross_judge"]
    assert cross["agree"] == 2
    assert cross["disagree"] == 0
    assert cross["deepseek_only_unresolved"] == 1
    assert cross["qwen_only_unresolved"] == 0
    assert cross["both_unresolved"] == 0
    assert cross["consensus_comparable"] == 2
    assert cross["raw_order_comparable"] == 6
    assert cross["raw_order_agreement"] == 5
    assert cross["raw_order_disagreement"] == 1
    assert cross["raw_order_disagreements"] == [
        {
            "pair_id": "p1",
            "order": "BA",
            "deepseek": "A",
            "qwen": "tie",
        }
    ]


def test_comparison_rejects_mismatched_blind_sources():
    deepseek = make_report([
        make_result("p1", "tie", "tie", "tie"),
        make_result("p2", "B", "B", "A"),
        make_result("p3", "A", "A", "B"),
    ])
    qwen = make_report(
        [
            make_result("p1", "tie", "tie", "tie"),
            make_result("p2", "B", "B", "A"),
            make_result("p3", "A", "A", "B"),
        ],
        blind_sha256="different-hash",
    )

    with pytest.raises(AssertionError):
        build_comparison(
            make_blind(),
            deepseek,
            qwen,
            "test-hash",
        )

def test_partial_last_run_does_not_invalidate_complete_saved_pairs():
    deepseek = make_report([
        make_result("p1", "tie", "tie", "tie"),
        make_result("p2", "B", "B", "A"),
        make_result("p3", "A", "A", "B"),
    ])
    deepseek["status"] = "partial"

    qwen = make_report([
        make_result("p1", "tie", "tie", "tie"),
        make_result("p2", "B", "B", "A"),
        make_result("p3", "A", "A", "B"),
    ])

    report = build_comparison(
        make_blind(),
        deepseek,
        qwen,
        "test-hash",
    )

    assert report["summary"]["total_pairs"] == 3
    assert (
        report["metadata"]["judges"]["deepseek"]["last_run_status"]
        == "partial"
    )

