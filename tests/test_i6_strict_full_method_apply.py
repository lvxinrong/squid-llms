import runpy
from pathlib import Path

import pytest


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_strict_full_method_apply.py"
))


def test_method_validation_ignores_braces_in_java_literals():
    buggy = "public void demo() {\n    int value = 1;\n}"
    candidate = (
        "public void demo() {\n"
        "    String text = \"}\";\n"
        "    char brace = '{';\n"
        "    // }\n"
        "    /* { */\n"
        "}"
    )

    result = namespace["validate_method_response"](
        candidate,
        buggy,
    )

    assert result["method_end"] == len(candidate)


def test_method_validation_rejects_extra_helper():
    buggy = "public void demo() {\n    int value = 1;\n}"
    candidate = (
        "public void demo() {\n    int value = 2;\n}\n"
        "public void helper() {\n}\n"
    )

    with pytest.raises(ValueError, match="额外声明"):
        namespace["validate_method_response"](
            candidate,
            buggy,
        )


def test_method_validation_rejects_different_signature():
    buggy = "public void demo() {\n    int value = 1;\n}"
    candidate = (
        "public void demo(int value) {\n"
        "    int other = value;\n"
        "}"
    )

    with pytest.raises(ValueError, match="签名匹配"):
        namespace["validate_method_response"](
            candidate,
            buggy,
        )


def test_exact_replacement_preserves_raw_candidate_and_affixes():
    buggy_method = (
        "public void demo() {\n"
        "    int value = 1;\n"
        "}"
    )
    original = (
        "class Demo {\n"
        + buggy_method
        + "\n    // keep this suffix\n}"
    )
    candidate = (
        "public void demo() {\n"
        "    int value = 2;\n"
        "}\n"
    )

    patched = namespace["replace_exact_method"](
        original,
        buggy_method,
        candidate,
    )

    assert candidate in patched
    assert patched.startswith("class Demo {\n")
    assert patched.endswith("\n    // keep this suffix\n}")
