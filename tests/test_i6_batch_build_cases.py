import runpy
from pathlib import Path


root = Path(__file__).resolve().parents[1]

builder_namespace = runpy.run_path(str(
    root / "scripts/i6_build_case.py"
))
audit_namespace = runpy.run_path(str(
    root / "scripts/i6_patch_audit.py"
))
namespace = runpy.run_path(str(
    root / "scripts/i6_batch_build_cases.py"
))

extract_java_block = builder_namespace["extract_java_block"]
method_declarations = audit_namespace["method_declarations"]
choose_method_marker = namespace["choose_method_marker"]


def test_choose_method_marker_uses_hunk_intersection():
    lines = [
        "class Demo {",
        "    public static int first() {",
        "        return 1;",
        "    }",
        "    public static int second() {",
        "        return 2;",
        "    }",
        "}",
    ]
    file_audit = {
        "changed_methods": ["method:second"],
        "hunks": [{"old_start": 6, "old_count": 1}],
    }

    marker = choose_method_marker(
        file_audit,
        lines,
        extract_java_block,
        method_declarations,
    )

    assert marker == "public static int second() {"
