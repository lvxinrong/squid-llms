import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_java_basics_reference.py"
))


def task():
    return {
        "id": "J0-test",
        "category": "test",
        "signature": "public static int add(int a, int b)",
        "specification": "返回两个 int 参数的和。",
        "public_example": "JavaBasics.add(2, 3) == 5",
        "reference_method": (
            "public static int add(int a, int b) {\n"
            "    return a + b;\n"
            "}"
        ),
        "public_checks": [
            "JavaBasics.add(2, 3) == 5",
        ],
        "private_checks": [
            "JavaBasics.add(1, 1) == 2",
        ],
    }


def test_reference_method_passes():
    result = namespace["evaluate_reference"](task())

    assert result["status"] == "passed"


def test_wrong_model_candidate_fails():
    wrong = (
        "public static int add(int a, int b) {\n"
        "    return a - b;\n"
        "}"
    )

    result = namespace["evaluate_method"](
        task(),
        wrong,
        "original",
    )

    assert result["status"] == "assertion_failed"
