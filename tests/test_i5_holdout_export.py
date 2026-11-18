import json
import runpy
from collections import Counter
from pathlib import Path

import pytest


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i5_holdout_export.py"
))
build_candidates = namespace["build_candidates"]
build_documents = namespace["build_documents"]
assert_safe_to_export = namespace["assert_safe_to_export"]


def test_holdout_has_ten_unique_tasks_and_balanced_conditions():
    candidates = build_candidates()
    blind, key = build_documents(candidates)

    assert len(blind["pairs"]) == 10
    assert len({
        pair["user"]
        for pair in blind["pairs"]
    }) == 10

    conditions = Counter(
        pair["condition"]
        for pair in key["pairs"]
    )
    assert conditions == {
        "correctness_gate": 2,
        "informative_length": 2,
        "padded_verbosity": 2,
        "equivalent": 2,
        "instruction_following": 2,
    }


def test_blind_export_is_deterministic_and_hides_design():
    candidates = build_candidates()
    first, first_key = build_documents(candidates)
    second, second_key = build_documents(candidates)

    assert first == second
    assert first_key == second_key

    public_text = json.dumps(first, ensure_ascii=False)
    assert "design_hypothesis" not in public_text
    assert "preferred_role" not in public_text
    assert "wrong_verbose" not in public_text
    assert "informative_length" not in public_text

    for pair in first["pairs"]:
        assert pair["human"] == {
            "winner": None,
            "reason_codes": [],
            "notes": "",
        }


def test_holdout_export_guard_protects_labels(tmp_path):
    path = tmp_path / "holdout.json"
    blind, _ = build_documents(build_candidates())

    path.write_text(
        json.dumps(blind),
        encoding="utf-8",
    )
    assert_safe_to_export(path)

    blind["pairs"][0]["human"]["winner"] = "A"
    path.write_text(
        json.dumps(blind),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        assert_safe_to_export(path)
