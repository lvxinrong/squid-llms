import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_build_case.py"
))

extract_java_block = namespace["extract_java_block"]


def test_extract_java_block_stops_at_method_end():
    lines = [
        "class Demo {",
        "    public static int first(int x) {",
        "        if (x > 0) {",
        "            return x;",
        "        }",
        "        return 0;",
        "    }",
        "    public static int second() {",
        "        return 2;",
        "    }",
        "}",
    ]

    result = extract_java_block(
        lines,
        "public static int first",
    )

    assert result["start_line"] == 2
    assert result["end_line"] == 7
    assert "return x" in result["text"]
    assert "return 0" in result["text"]
    assert "second" not in result["text"]

