import runpy
from pathlib import Path

import pytest


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_structured_sft_data.py"
))
reference_to_edit = namespace["reference_to_edit"]


def test_reference_patch_becomes_minimal_replace_edit():
    patch = """diff --git a/Demo.java b/Demo.java
@@ -1 +1 @@
-    if (str.startsWith("0x")) {
+    if (str.startsWith("0x") || str.startsWith("0X")) {
"""
    source = '    if (str.startsWith("0x")) {'

    edit = reference_to_edit(patch, "createNumber", source)

    assert edit == {
        "action": "replace",
        "target_method": "createNumber",
        "old_text": '    if (str.startsWith("0x")) {',
        "new_text": (
            '    if (str.startsWith("0x") || '
            'str.startsWith("0X")) {'
        ),
    }


def test_reference_edit_discards_context_outside_method():
    patch = """@@ -1,5 +1,5 @@
     /**
      */
-    int value = 1;
+    int value = 2;
     int other = 3;
     /**
"""
    source = "    int value = 1;\n    int other = 3;"

    edit = reference_to_edit(patch, "demo", source)

    assert edit["old_text"] == "    int value = 1;"
    assert edit["new_text"] == "    int value = 2;"


def test_pure_insert_uses_unique_anchor():
    patch = """@@ -1,2 +1,3 @@
 class Demo {
+    int added = 1;
     int other = 3;
"""
    source = "class Demo {\n    int other = 3;"

    edit = reference_to_edit(patch, "demo", source)

    assert edit["old_text"] == "class Demo {"
    assert edit["new_text"] == (
        "class Demo {\n"
        "    int added = 1;"
    )


def test_duplicate_change_expands_to_unique_context():
    patch = """@@ -1,2 +1,2 @@
     int left = 0;
-    int value = 1;
+    int value = 2;
"""
    source = (
        "    int value = 1;\n"
        "    int left = 0;\n"
        "    int value = 1;"
    )

    edit = reference_to_edit(patch, "demo", source)

    assert edit["old_text"] == (
        "    int left = 0;\n"
        "    int value = 1;"
    )
    assert edit["new_text"] == (
        "    int left = 0;\n"
        "    int value = 2;"
    )


def test_missing_change_is_rejected():
    patch = """@@ -1 +1 @@
-    int value = 1;
+    int value = 2;
"""

    with pytest.raises(ValueError, match="精确匹配"):
        reference_to_edit(
            patch,
            "demo",
            "    int value = 9;",
        )


def test_multiple_hunks_remain_unsupported():
    patch = """@@ -1 +1 @@
-old
+new
@@ -3 +3 @@
-before
+after
"""

    with pytest.raises(ValueError, match="单个连续修改块"):
        reference_to_edit(patch, "demo", "old\nbefore")
