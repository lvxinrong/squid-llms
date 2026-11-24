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
