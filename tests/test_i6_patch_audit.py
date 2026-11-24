import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_patch_audit.py"
))

parse_hunks = namespace["parse_hunks"]
locate_methods = namespace["locate_methods"]


def test_parse_hunks_preserves_old_positions():
    diff = """@@ -10,2 +10,3 @@
@@ -25 +26,4 @@
"""

    assert parse_hunks(diff) == [
        {"old_start": 10, "old_count": 2},
        {"old_start": 25, "old_count": 1},
    ]


def test_changed_methods_use_method_boundaries():
    lines = [
        "class Demo {",
        "    public int first() {",
        "        return 1;",
        "    }",
        "    public int second() {",
        "        return 2;",
        "    }",
        "}",
    ]

    result = locate_methods(
        lines,
        [{"old_start": 3, "old_count": 1}],
    )

    assert result == ["method:first"]
