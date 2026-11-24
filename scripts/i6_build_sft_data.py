import argparse
import json
import re
import runpy
from pathlib import Path

from transformers import AutoTokenizer

from squid.data.chat_sft import encode_qwen_multiturn


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"


def build_entry(case, fixed_method):
    signature = case["buggy_source"].splitlines()[0].strip()
    match = re.search(r"(\w+)\s*\(", signature)
    assert match is not None
    target_method = match.group(1)

    task = f"""你是一名 Java 程序修复助手。请阅读 buggy 源码和触发测试，定位缺陷，并输出修复后的完整 {target_method} 方法。

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

    return {
        "instance_id": case["instance_id"],
        "data_role": case["data_role"],
        "target_method": target_method,
        "messages": [
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
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument(
        "--workroot",
        default=str(Path.home() / "experiments/i6/workdirs"),
    )
    args = parser.parse_args()

    train_paths = [Path(path) for path in args.train]
    val_paths = [Path(path) for path in args.val]

    all_ids = []
    for path in train_paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        assert case["data_role"] == "train_candidate"
        assert case["excluded_from_training"] is False
        all_ids.append(case["instance_id"])

    for path in val_paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        assert case["data_role"] == "validation_candidate"
        assert case["excluded_from_training"] is True
        all_ids.append(case["instance_id"])

    assert len(all_ids) == len(set(all_ids)), "train / val 存在实例重叠"

    namespace = runpy.run_path(str(
        ROOT / "scripts/i6_build_case.py"
    ))
    extract_java_block = namespace["extract_java_block"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    def build_dataset(paths):
        entries = []

        for path in paths:
            case = json.loads(path.read_text(encoding="utf-8"))
            fixed_file = (
                Path(args.workroot)
                / case["instance_id"].lower()
                / "fixed"
                / case["source_file"]
            )

            marker = case["buggy_source"].splitlines()[0].strip()
            fixed_lines = fixed_file.read_text(
                encoding="utf-8"
            ).splitlines()
            fixed_method = extract_java_block(
                fixed_lines,
                marker,
            )["text"]

            entry = build_entry(case, fixed_method)
            full_ids, targets = encode_qwen_multiturn(
                tokenizer,
                entry["messages"],
                enable_thinking=False,
            )

            entry["token_audit"] = {
                "full_tokens": len(full_ids),
                "supervised_tokens": sum(
                    target != -100
                    for target in targets
                ),
            }
            entries.append(entry)

            print(
                f"{case['instance_id']}: "
                f"full={len(full_ids)}, "
                f"supervised={entry['token_audit']['supervised_tokens']}"
            )

        return entries

    train_entries = build_dataset(train_paths)
    val_entries = build_dataset(val_paths)

    for output, entries in (
        (Path(args.train_output), train_entries),
        (Path(args.val_output), val_entries),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"output: {output}，samples={len(entries)}")

    print("boundary: 小批量对照数据，不用于证明未见项目泛化")


if __name__ == "__main__":
    main()
