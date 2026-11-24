import runpy
from pathlib import Path


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

    edit = reference_to_edit(patch, "createNumber")

    assert edit == {
        "action": "replace",
        "target_method": "createNumber",
        "old_text": '    if (str.startsWith("0x")) {',
        "new_text": (
            '    if (str.startsWith("0x") || '
            'str.startsWith("0X")) {'
        ),
    }

def test_reference_edit_preserves_contiguous_context():
    patch = """diff --git a/Demo.java b/Demo.java
@@ -1,4 +1,4 @@
 class Demo {
-    int value = 1;
+    int value = 2;
     int other = 3;
 }
"""

    edit = reference_to_edit(patch, "demo")

    assert edit["action"] == "replace"
    assert edit["old_text"] == (
        "class Demo {\n"
        "    int value = 1;\n"
        "    int other = 3;\n"
        "}"
    )
    assert edit["new_text"] == (
        "class Demo {\n"
        "    int value = 2;\n"
        "    int other = 3;\n"
        "}"
    )


def test_pure_insert_keeps_anchor_context():
    patch = """@@ -1,2 +1,3 @@
 class Demo {
+    int added = 1;
     int other = 3;
"""

    edit = reference_to_edit(patch, "demo")

    assert edit["action"] == "replace"
    assert edit["old_text"] == "class Demo {\n    int other = 3;"
    assert edit["new_text"] == (
        "class Demo {\n"
        "    int added = 1;\n"
        "    int other = 3;"
    )

