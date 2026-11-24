import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_structured_edit_baseline.py"
))
validate_edit = namespace["validate_edit"]


def case():
    return {
        "buggy_source": (
            "    public static Number createNumber(String str) {\n"
            "        if (str.startsWith(\"0x\")) {\n"
            "            return createInteger(str);\n"
            "        }\n"
            "        return null;\n"
            "    }"
        ),
    }


def test_valid_unique_local_edit_is_ready():
    result = validate_edit(
        case(),
        {
            "action": "replace",
            "target_method": "createNumber",
            "old_text": 'str.startsWith("0x")',
            "new_text": 'str.startsWith("0x") || str.startsWith("0X")',
        },
    )
    assert result == {"status": "ready_to_apply", "match_count": 1}


def test_duplicate_match_is_rejected():
    result = validate_edit(
        case(),
        {
            "action": "replace",
            "target_method": "createNumber",
            "old_text": "return",
            "new_text": "return",
        },
    )
    assert result["status"] == "no_op_edit"
