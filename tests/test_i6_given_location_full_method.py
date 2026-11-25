import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_given_location_full_method.py"
))


def case():
    return {
        "target_class": "Demo",
        "buggy_source": (
            "public void demo() {\n"
            "    int value = 1;\n"
            "}"
        ),
        "trigger_test": "DemoTest::testDemo",
        "test_source": "public void testDemo() {}",
    }


def test_default_prompt_preserves_verbatim_old_text():
    task = namespace["build_full_method_task"](
        case(),
        "demo",
        "    int value = 1;",
    )

    assert "[给定待修改原文]" in task
    assert "    int value = 1;" in task
    assert task.endswith(
        "[给定待修改原文]\n"
        "    int value = 1;"
    )


def test_line_location_prompt_does_not_repeat_old_text():
    task = namespace["build_full_method_task"](
        case(),
        "demo",
        "    int value = 1;",
        location_lines="2",
    )

    assert "[给定待修改位置]" in task
    assert "目标方法第 2 行" in task
    assert "    int value = 1;\n}" in task
    assert "[给定待修改原文]" not in task
    assert task.count("    int value = 1;") == 1
