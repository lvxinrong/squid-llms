import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_build_case.py"
))

find_java_file = namespace["find_java_file"]

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


def test_find_java_file_supports_legacy_source_layout(tmp_path):
    source = (
        tmp_path
        / "src/java/org/apache/commons/lang3/math/NumberUtils.java"
    )
    source.parent.mkdir(parents=True)
    source.write_text("class NumberUtils {}", encoding="utf-8")

    result = find_java_file(
        tmp_path,
        "org.apache.commons.lang3.math.NumberUtils",
        ("src/main/java", "src/java", "src"),
    )

    assert result == source


def test_find_java_file_supports_legacy_test_layout(tmp_path):
    test = tmp_path / "src/test/example/DemoTest.java"
    test.parent.mkdir(parents=True)
    test.write_text("class DemoTest {}", encoding="utf-8")

    result = find_java_file(
        tmp_path,
        "example.DemoTest",
        ("src/test/java", "src/test", "test"),
    )

    assert result == test


