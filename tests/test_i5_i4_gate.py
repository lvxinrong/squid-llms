import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i5_i4_gate.py"
))
build_gate = namespace["build_gate"]


def pair(effect):
    return {
        "effect_from_net_interval": effect,
        "bootstrap": {
            "observed": {
                "net_win_rate": 0,
            },
            "intervals": {
                "net_win_rate": {
                    "low": 0,
                    "high": 0,
                }
            },
        },
    }


def ci_report(cot_effect, dpo_effect, grpo_effect):
    return {
        "pairwise": {
            "original vs cot_sft_v2": pair(cot_effect),
            "original vs dpo_v1": pair(dpo_effect),
            "original vs grpo_one_group": pair(grpo_effect),
        }
    }


def coverage():
    return {
        "models": 4,
        "cases_per_model": 32,
        "aligned_indices": 32,
        "pairwise_comparisons": 6,
    }


def test_gate_maps_original_direction_to_candidate_status():
    gate = build_gate(
        ci_report("left_better", "left_worse", "inconclusive"),
        coverage(),
    )

    assert gate["candidates"]["cot_sft_v2"]["status"] == (
        "regression_detected"
    )
    assert gate["candidates"]["dpo_v1"]["status"] == (
        "improvement_detected"
    )
    assert gate["candidates"]["grpo_one_group"]["status"] == (
        "inconclusive"
    )
    assert gate["regression_detected"] is True
    assert gate["evidence_status"] == "regression_detected"


def test_all_inconclusive_is_not_artifact_failure():
    gate = build_gate(
        ci_report("inconclusive", "inconclusive", "inconclusive"),
        coverage(),
    )

    assert gate["artifact_status"] == "pass"
    assert gate["regression_detected"] is False
    assert gate["evidence_status"] == "inconclusive"


def test_all_candidates_must_improve_for_overall_improvement():
    gate = build_gate(
        ci_report("left_worse", "left_worse", "inconclusive"),
        coverage(),
    )

    assert gate["regression_detected"] is False
    assert gate["evidence_status"] == "inconclusive"
