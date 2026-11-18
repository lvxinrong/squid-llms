import runpy
from pathlib import Path

import pytest


script_path = (
    Path(__file__).resolve().parents[1]
    / "scripts/i4_behavior_baseline.py"
)
namespace = runpy.run_path(str(script_path))
evaluate_case = namespace["evaluate_case"]
is_previous_assistant_copy = namespace[
    "is_previous_assistant_copy"
]

@pytest.mark.parametrize(
    "expected,response,answer_correct,format_compliant,passed",
    [
        ("30", "24", False, True, False),
        ("600", "240", False, True, False),
        ("912", "123 + 789 = 912", True, False, False),
        ("912", "912", True, True, True),
    ],
)
def test_numeric_answer_and_format_are_independent(
    expected,
    response,
    answer_correct,
    format_compliant,
    passed,
):
    case = {"expected_exact": expected}

    actual = evaluate_case(case, response)

    assert actual[:3] == (
        answer_correct,
        format_compliant,
        passed,
    )

@pytest.mark.parametrize(
    "response,expected",
    [
        ("我记住了，你喜欢蓝色。", True),
        ("你喜欢蓝色。", False),
        ("24", False),
    ],
)
def test_previous_assistant_copy_detection(response, expected):
    case = {
        "history": [
            ["我喜欢蓝色。", "我记住了，你喜欢蓝色。"]
        ]
    }

    assert is_previous_assistant_copy(case, response) == expected
