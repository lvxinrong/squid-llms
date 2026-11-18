import runpy
from pathlib import Path

import torch


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i0_numerical_alignment_report.py"
))
tensor_metrics = namespace["tensor_metrics"]
logits_metrics = namespace["logits_metrics"]


def test_tensor_metrics_reports_difference_and_tolerance():
    our = torch.tensor([1.0, 2.0])
    official = torch.tensor([1.0, 2.00001])

    result = tensor_metrics(our, official)

    assert result["allclose"] is True
    assert result["our_finite"] is True
    assert result["official_finite"] is True
    assert result["max_abs_difference"] > 0


def test_logits_metrics_checks_next_token_and_finite_values():
    our = torch.tensor([[[0.0, 1.0, 3.0]]])
    official = torch.tensor([[[0.0, 1.0, 3.0]]])

    result = logits_metrics(our, official)

    assert result["passed"] is True
    assert result["next_token_match"] is True
    assert result["our_next_token"] == 2
    assert result["official_next_token"] == 2
