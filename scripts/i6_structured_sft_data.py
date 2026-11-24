import argparse
import json
import runpy
from pathlib import Path

from transformers import AutoTokenizer

from squid.data.chat_sft import encode_qwen_multiturn


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"


def reference_to_edit(reference_patch, target_method, buggy_source):
    hunk = []
    hunk_count = 0
    in_hunk = False

    for line in reference_patch.splitlines():
        if line.startswith("@@"):
            hunk_count += 1
            if in_hunk:
                break
            in_hunk = True
            continue

        if not in_hunk:
            continue
        if line.startswith(("diff --git", "index ", "---", "+++")):
            continue

        if line.startswith(" "):
            hunk.append((" ", line[1:]))
        elif line.startswith("-"):
            hunk.append(("-", line[1:]))
        elif line.startswith("+"):
            hunk.append(("+", line[1:]))

    if hunk_count != 1:
        raise ValueError("第一版只支持单个连续修改块")

    changed = [
        index
        for index, (kind, _) in enumerate(hunk)
        if kind in {"-", "+"}
    ]
    if not changed:
        raise ValueError("没有可转换的修改")

    first_change = changed[0]
    last_change = changed[-1] + 1
    max_left = first_change
    max_right = len(hunk) - last_change

    for extra in range(max_left + max_right + 1):
        for left in range(min(extra, max_left), -1, -1):
            right = extra - left
            if right > max_right:
                continue

            start = first_change - left
            end = last_change + right
            selected = hunk[start:end]

            old_text = "\n".join(
                text
                for kind, text in selected
                if kind != "+"
            )
            new_text = "\n".join(
                text
                for kind, text in selected
                if kind != "-"
            )

            if not old_text or not new_text:
                continue

            if buggy_source.count(old_text) == 1:
                return {
                    "action": "replace",
                    "target_method": target_method,
                    "old_text": old_text,
                    "new_text": new_text,
                }

    core_old = "\n".join(
        text
        for kind, text in hunk[first_change:last_change]
        if kind != "+"
    )
    if core_old and buggy_source.count(core_old) == 0:
        raise ValueError("修改核心无法在目标方法中精确匹配")

    raise ValueError("无法构造非空且唯一的局部编辑")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))

    target_method = re_target_method(case["buggy_source"])

    edit = reference_to_edit(
        key["reference_patch"],
        target_method,
        case["buggy_source"],
    )

    namespace = runpy.run_path(str(
        ROOT / "scripts/i6_structured_edit_baseline.py"
    ))
    validation = namespace["validate_edit"](case, edit)
    assert validation["status"] == "ready_to_apply", validation

    task = f"""你是一名 Java 程序修复助手。请定位缺陷，并输出一个最小局部编辑。

只输出一个 JSON 对象，不要 Markdown 或解释：
{{
  "action": "replace",
  "target_method": "{target_method}",
  "old_text": "输入源码中需要修改的连续原文",
  "new_text": "用于替换 old_text 的最小修复代码"
}}

要求：
- old_text 必须从输入源码中精确复制。
- old_text 在目标方法中必须只出现一次。
- new_text 只修复缺陷，不做无关重写。

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
            "content": json.dumps(
                edit,
                ensure_ascii=False,
            ),
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    full_ids, targets = encode_qwen_multiturn(
        tokenizer,
        messages,
        enable_thinking=False,
    )

    entry = {
        "instance_id": case["instance_id"],
        "data_role": "structured_train_smoke",
        "messages": messages,
        "edit": edit,
        "token_audit": {
            "full_tokens": len(full_ids),
            "supervised_tokens": sum(
                target != -100
                for target in targets
            ),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([entry], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(edit, ensure_ascii=False, indent=2))
    print(json.dumps(entry["token_audit"], ensure_ascii=False))
    print(f"output: {output}")


def re_target_method(source):
    import re
    match = re.search(
        r"(?:public|protected|private)\s+"
        r"(?:static\s+)?(?:final\s+)?"
        r"[\w$<>\[\],.?]+\s+(\w+)\s*\(",
        source,
    )
    if not match:
        raise ValueError("无法识别目标方法")
    return match.group(1)


if __name__ == "__main__":
    main()
