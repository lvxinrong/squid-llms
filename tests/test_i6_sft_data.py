import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_build_sft_data.py"
))
build_entry = namespace["build_entry"]


def test_training_entry_uses_fixed_method_only_as_assistant_target():
    case = {
        "instance_id": "Lang-5",
        "data_role": "train_candidate",
        "target_class": "org.apache.commons.lang3.LocaleUtils",
        "trigger_test": "LocaleUtilsTest::testLang865",
        "buggy_source": (
            "    public static Locale toLocale(final String str) {\n"
            "        return null;\n"
            "    }"
        ),
        "test_source": "测试内容",
    }
    fixed_method = (
        "public static Locale toLocale(final String str) {\n"
        "    // 修复\n"
        "}"
    )

    entry = build_entry(case, fixed_method)

    assert entry["target_method"] == "toLocale"
    assert entry["messages"][2]["content"] == fixed_method
    assert fixed_method not in entry["messages"][1]["content"]
    assert entry["data_role"] == "train_candidate"
