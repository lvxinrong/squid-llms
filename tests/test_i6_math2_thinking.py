import runpy
from pathlib import Path

import pytest


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_math2_thinking.py"
))


def test_non_thinking_response_is_unchanged():
    raw = "public void demo() {}"
    response, reasoning = namespace["extract_final_response"](
        raw,
        thinking=False,
    )

    assert response == raw
    assert reasoning is None


def test_thinking_response_uses_text_after_close_tag():
    raw = "analysis</think>\npublic void demo() {}"
    response, reasoning = namespace["extract_final_response"](
        raw,
        thinking=True,
    )

    assert response == "\npublic void demo() {}"
    assert reasoning == "analysis"


def test_missing_thinking_close_tag_is_rejected():
    with pytest.raises(ValueError, match="必须且只能"):
        namespace["extract_final_response"](
            "analysis without close tag",
            thinking=True,
        )
