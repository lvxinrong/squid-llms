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


def test_insert_after_preserves_anchor_and_new_content():
    result = validate_edit(
        case(),
        {
            "action": "insert_after",
            "target_method": "createNumber",
            "old_text": 'str.startsWith("0x")',
            "new_text": "// 补充大写前缀处理",
        },
    )
    assert result["status"] == "ready_to_apply"


def test_delete_allows_empty_new_text():
    result = validate_edit(
        case(),
        {
            "action": "delete",
            "target_method": "createNumber",
            "old_text": 'str.startsWith("0x")',
            "new_text": "",
        },
    )
    assert result["status"] == "ready_to_apply"


def test_replace_still_rejects_no_op():
    result = validate_edit(
        case(),
        {
            "action": "replace",
            "target_method": "createNumber",
            "old_text": 'str.startsWith("0x")',
            "new_text": 'str.startsWith("0x")',
        },
    )
    assert result["status"] == "no_op_edit"


def test_structured_task_contains_contract_and_all_inputs():
    task_case = {
        **case(),
        "target_class": "example.NumberUtils",
        "trigger_test": "example.NumberUtilsTest::testCreateNumber",
        "test_source": "public void testCreateNumber() {}",
    }

    task = namespace["build_structured_task"](
        task_case,
        "createNumber",
    )

    assert "不要输出完整方法" in task
    assert "不能修改测试" not in task
    assert "不要修改测试" in task
    assert task_case["target_class"] in task
    assert task_case["buggy_source"] in task
    assert task_case["trigger_test"] in task
    assert task_case["test_source"] in task
