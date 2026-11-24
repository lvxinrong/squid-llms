import json
import runpy
from pathlib import Path

from transformers import AutoTokenizer

from squid.data.chat_sft import encode_qwen_multiturn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"

def main():
    case_path = Path.home() / "experiments/i6/cases/lang-5-case.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))

    assert case["data_role"] == "train_candidate"
    assert case["excluded_from_training"] is False
    assert case["instance_id"] == "Lang-5"

    workdir = Path.home() / "experiments/i6/workdirs/lang-5"
    fixed_file = workdir / "fixed" / case["source_file"]

    namespace = runpy.run_path(str(
        ROOT / "scripts/i6_build_case.py"
    ))
    extract_java_block = namespace["extract_java_block"]

    fixed_lines = fixed_file.read_text(
        encoding="utf-8"
    ).splitlines()
    fixed_method = extract_java_block(
        fixed_lines,
        "public static Locale toLocale",
    )["text"]

    assert fixed_method != case["buggy_source"]

    task = f"""你是一名 Java 程序修复助手。请阅读 buggy 源码和触发测试，定位缺陷，并输出修复后的完整 toLocale 方法。

要求：
1. 只输出完整的 Java 方法源码，不要输出 unified diff、解释或 Markdown 代码围栏。
2. 不要删除或截断方法，不要改变方法签名。
3. 保留与缺陷无关的原有逻辑，只做最小修复。
4. 修复必须满足触发测试中的类型与格式要求。
5. 不要复述题目。

[目标类]
{case["target_class"]}

[Buggy 方法源码]
{case["buggy_source"]}

[触发测试]
{case["trigger_test"]}

{case["test_source"]}"""

    messages = [
        {
            "role": "system",
            "content": "你是一个严谨的 Java 程序修复助手。",
        },
        {
            "role": "user",
            "content": task,
        },
        {
            "role": "assistant",
            "content": fixed_method,
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    full_ids, targets = encode_qwen_multiturn(
        tokenizer,
        messages,
    )
    supervised_tokens = sum(
        target != -100
        for target in targets
    )

    assert supervised_tokens > 0
    assert fixed_method in tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    output_path = (
        ROOT / "data/instruct/i6_lang5_smoke.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([
            {
                "instance_id": "Lang-5",
                "data_role": "train_smoke",
                "messages": messages,
            }
        ], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"full_tokens: {len(full_ids)}")
    print(f"supervised_tokens: {supervised_tokens}")
    print(f"fixed_method_characters: {len(fixed_method)}")
    print(f"output: {output_path}")
    print("boundary: 单样本训练冒烟，不是独立验证集或领域能力证明")


if __name__ == "__main__":
    main()