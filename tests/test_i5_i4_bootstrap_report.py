import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i5_i4_bootstrap_report.py"
))

classify_net_effect = namespace["classify_net_effect"]
compare_with_bootstrap = namespace["compare_with_bootstrap"]


def case(index, passed):
    return {
        "index": index,
        "passed": passed,
    }


def test_net_interval_classification():
    assert classify_net_effect({
        "low": 0.1,
        "high": 0.3,
    }) == "left_better"

    assert classify_net_effect({
        "low": -0.3,
        "high": -0.1,
    }) == "left_worse"

    assert classify_net_effect({
        "low": -0.1,
        "high": 0.2,
    }) == "inconclusive"

    assert classify_net_effect({
        "low": 0,
        "high": 0.2,
    }) == "inconclusive"

    assert classify_net_effect(None) == "unknown"


def test_comparison_attaches_bootstrap_to_existing_wlt():
    left = [case(0, True), case(1, True)]
    right = [case(0, False), case(1, False)]

    result = compare_with_bootstrap(
        left,
        right,
        n_resamples=100,
        seed=7,
        confidence=0.95,
    )

    assert result["wins"] == 2
    assert result["losses"] == 0
    assert result["ties"] == 0
    assert result["win_rate_all"] == 1
    assert result["effect_from_net_interval"] == "left_better"
    assert result["bootstrap"]["n_resamples"] == 100
    assert result["bootstrap"]["intervals"]["net_win_rate"] == {
        "low": 1,
        "high": 1,
    }
