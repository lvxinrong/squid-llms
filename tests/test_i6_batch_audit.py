import runpy
from pathlib import Path
import subprocess


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_batch_audit.py"
))
classify_bug = namespace["classify_bug"]


def test_batch_classification_keeps_multi_location_boundaries():
    assert classify_bug([
        {"classification": "single_method_candidate"}
    ]) == "single_method_candidate"

    assert classify_bug([
        {"classification": "contains_class_level_change"}
    ]) == "multi_location"

    assert classify_bug([
        {"classification": "single_method_candidate"},
        {"classification": "single_method_candidate"},
    ]) == "multi_file"


def test_checkout_failure_is_returned_for_batch_entry(
    tmp_path,
    monkeypatch,
):
    def fail_checkout(command, check):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(
        namespace["subprocess"],
        "run",
        fail_checkout,
    )

    error = namespace["try_checkout_missing"](
        tmp_path,
        "Math",
        "7",
    )

    assert error is not None
    assert error.startswith("checkout 7b failed:")
