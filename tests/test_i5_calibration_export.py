import json
import runpy
from collections import Counter
from pathlib import Path
import random
import pytest



script_path = (
    Path(__file__).resolve().parents[1]
    / "scripts/i5_calibration_export.py"
)
namespace = runpy.run_path(str(script_path))

collect_candidates = namespace["collect_candidates"]
select_candidates = namespace["select_candidates"]
build_blind_pair = namespace["build_blind_pair"]
assert_safe_to_export = namespace["assert_safe_to_export"]



def make_case(index, passed, response):
    return {
        "index": index,
        "category": "single_arithmetic",
        "history": [],
        "user": "1 + 1 等于多少？",
        "expected_answer": "2",
        "response": response,
        "passed": passed,
        "generated_tokens": len(response),
    }


def test_collect_candidates_covers_three_rule_strata():
    cases_by_model = {
        "original": {
            0: make_case(0, True, "2"),
            1: make_case(1, True, "2"),
            2: make_case(2, False, "3"),
        },
        "dpo_v1": {
            0: make_case(0, False, "3"),
            1: make_case(1, True, "2"),
            2: make_case(2, False, "4"),
        },
    }

    candidates = collect_candidates(cases_by_model)
    strata = Counter(
        candidate["stratum"]
        for candidate in candidates
    )

    assert strata == {
        "rule_disagreement": 1,
        "both_passed": 1,
        "both_failed": 1,
    }


def test_selection_is_deterministic():
    targets = {
        "rule_disagreement": 1,
        "both_passed": 1,
        "both_failed": 1,
    }
    candidates = [
        {
            "candidate_id": f"{stratum}-{index}",
            "stratum": stratum,
            "case_index": stratum_number * 10 + index,
        }
        for stratum_number, stratum in enumerate(targets)
        for index in range(4)
    ]

    first = select_candidates(candidates, targets=targets, seed=7)
    second = select_candidates(candidates, targets=targets, seed=7)

    assert first == second
    assert len(first) == 3
    assert Counter(
        candidate["stratum"]
        for candidate in first
    ) == targets


def test_public_pair_hides_model_identity_and_rule_result():
    candidate = {
        "case_index": 7,
        "category": "single_arithmetic",
        "history": [],
        "user": "1 + 1 等于多少？",
        "expected_answer": "2",
        "stratum": "rule_disagreement",
        "left_model": "original",
        "right_model": "dpo_v1",
        "left": make_case(7, True, "答案为 2"),
        "right": make_case(7, False, "答案为 3"),
    }

    public, private = build_blind_pair(
        candidate,
        "i5-test-01",
        random.Random(7),
    )

    public_text = json.dumps(public, ensure_ascii=False)

    assert public["pair_id"] == "i5-test-01"
    assert set(public["responses"].values()) == {
        "答案为 2",
        "答案为 3",
    }
    assert public["human"] == {
        "winner": None,
        "reason_codes": [],
        "notes": "",
    }
    assert "original" not in public_text
    assert "dpo_v1" not in public_text
    assert "passed" not in public_text
    assert "generated_tokens" not in public_text

    assert {
        side["model"]
        for side in private["sides"].values()
    } == {"original", "dpo_v1"}


def test_selection_deduplicates_case_index_across_strata():
    targets = {
        "rule_disagreement": 1,
        "both_passed": 1,
    }
    candidates = [
        {
            "candidate_id": "disagreement-0",
            "stratum": "rule_disagreement",
            "case_index": 0,
        },
        {
            "candidate_id": "passed-0",
            "stratum": "both_passed",
            "case_index": 0,
        },
        {
            "candidate_id": "passed-1",
            "stratum": "both_passed",
            "case_index": 1,
        },
    ]

    selected = select_candidates(
        candidates,
        targets=targets,
        seed=7,
    )

    assert len(selected) == 2
    assert {
        candidate["case_index"]
        for candidate in selected
    } == {0, 1}
    assert {
        candidate["candidate_id"]
        for candidate in selected
    } == {"disagreement-0", "passed-1"}

def test_export_guard_protects_human_labels(tmp_path):
    blind_path = tmp_path / "blind.json"
    document = {
        "status": "unlabeled",
        "pairs": [
            {
                "human": {
                    "winner": None,
                }
            }
        ],
    }

    blind_path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    assert_safe_to_export(blind_path)

    document["pairs"][0]["human"]["winner"] = "A"
    blind_path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        assert_safe_to_export(blind_path)

    document["status"] = "labeled"
    document["pairs"][0]["human"]["winner"] = None
    blind_path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        assert_safe_to_export(blind_path)
