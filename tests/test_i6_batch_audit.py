import runpy
from pathlib import Path


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
