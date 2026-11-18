import json
import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i5_regression_coverage.py"
))
build_coverage_report = namespace["build_coverage_report"]


def test_needs_export_marks_coverage_incomplete(tmp_path):
    report = build_coverage_report(
        [
            {
                "stage": "I0",
                "name": "alignment",
                "coverage_class": "needs_export",
                "paths": [],
                "required_fields": [],
            }
        ],
        root=tmp_path,
    )

    assert report["artifact_status"] == "pass"
    assert report["coverage_status"] == "incomplete"
    assert report["entries"][0]["artifact_status"] == "not_applicable"


def test_valid_registered_artifact_is_hashed(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps({"bpb": 4.9}),
        encoding="utf-8",
    )

    report = build_coverage_report(
        [
            {
                "stage": "I1",
                "name": "bpb",
                "coverage_class": "single_observation",
                "paths": ["report.json"],
                "required_fields": ["bpb"],
            }
        ],
        root=tmp_path,
    )

    assert report["artifact_status"] == "pass"
    assert report["coverage_status"] == "complete_for_registered_scope"
    assert len(report["entries"][0]["artifacts"][0]["sha256"]) == 64


def test_missing_required_field_fails_artifact_validation(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("{}", encoding="utf-8")

    report = build_coverage_report(
        [
            {
                "stage": "I1",
                "name": "bpb",
                "coverage_class": "single_observation",
                "paths": ["report.json"],
                "required_fields": ["bpb"],
            }
        ],
        root=tmp_path,
    )

    assert report["artifact_status"] == "fail"
    assert report["summary"]["invalid_entries"] == 1
