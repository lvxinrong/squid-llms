import json
import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i5_regression.py"
))
summarize_training_runs = namespace["summarize_training_runs"]
determine_overall = namespace["determine_overall"]


def test_training_run_summary_reports_median_range_and_finite_losses(tmp_path):
    for index, throughput in enumerate((10.0, 20.0, 30.0), 1):
        (tmp_path / f"run-{index}.json").write_text(
            json.dumps({
                "tokens_per_second": throughput,
                "losses": [1.0, 0.5],
            }),
            encoding="utf-8",
        )

    result = summarize_training_runs(
        tmp_path,
        ["run-1.json", "run-2.json", "run-3.json"],
    )

    assert result["runs"] == 3
    assert result["tokens_per_second"] == {
        "min": 10.0,
        "median": 20.0,
        "max": 30.0,
    }
    assert result["all_losses_finite"] is True


def test_overall_status_does_not_hide_inconclusive_capability():
    checks = {"alignment": True}

    assert determine_overall(
        "pass",
        checks,
        False,
        "inconclusive",
    ) == ("artifacts_valid_capability_inconclusive", 0)

    assert determine_overall(
        "pass",
        checks,
        True,
        "inconclusive",
    ) == ("regression_detected", 2)

    assert determine_overall(
        "fail",
        checks,
        False,
        "inconclusive",
    ) == ("failed", 1)

