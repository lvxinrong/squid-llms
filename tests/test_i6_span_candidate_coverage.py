import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_span_candidate_coverage.py"
))


def test_contiguous_candidates_cover_exact_reference_span():
    source = (
        "class Demo {\n"
        "    int first = 1;\n"
        "    int second = 2;\n"
        "    int third = 3;\n"
        "}"
    )
    spans = namespace["build_spans"](source, max_lines=16)
    target = "    int second = 2;\n    int third = 3;"

    matches = [
        span
        for span in spans
        if span["text"] == target
    ]

    assert len(spans) == 15
    assert len(matches) == 1
    assert matches[0]["start_line"] == 3
    assert matches[0]["end_line"] == 4
