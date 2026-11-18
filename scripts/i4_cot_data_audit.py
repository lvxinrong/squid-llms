import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from squid.data.chat_sft import QwenMultiTurnDataset


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"

EXPRESSION_RE = re.compile(r"(\d+)\s*([+×*\-])\s*(\d+)")
EQUATION_RE = re.compile(
    r"(\d+)\s*([+×*\-])\s*(\d+)\s*=\s*(\d+)"
)


def extract_expressions(text):
    expressions = set()

    for left, operator, right in EXPRESSION_RE.findall(text):
        if operator == "*":
            operator = "×"
        expressions.add((int(left), operator, int(right)))

    return expressions


def check_reasoning(reasoning, path, sample_index):
    lines = [
        line.strip()
        for line in reasoning.splitlines()
        if line.strip()
    ]
    assert 1 <= len(lines) <= 3, (
        f"{path.name}[{sample_index}] 推理行数不是 1～3 行"
    )

    for line in lines:
        assert not any(
            text in line
            for text in ("不对", "可能", "大概", "我猜")
        ), f"{path.name}[{sample_index}] 含犹豫或纠错措辞：{line}"

    equation_count = 0
    unparsed = []

    for line in lines:
        matches = EQUATION_RE.findall(line)
        if "=" in line and not matches:
            unparsed.append(line)

        for left, operator, right, result in matches:
            left = int(left)
            right = int(right)
            result = int(result)

            if operator in {"×", "*"}:
                expected = left * right
            elif operator == "+":
                expected = left + right
            else:
                expected = left - right

            assert expected == result, (
                f"{path.name}[{sample_index}] 算式错误：{line}"
            )
            equation_count += 1

    assert not unparsed, (
        f"{path.name}[{sample_index}] 存在未解析等式：{unparsed}"
    )
    return equation_count


def audit_file(path, expected_total, tokenizer):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == expected_total

    normalized = []
    categories = Counter()
    expressions = set()
    supervised_counts = []
    equation_count = 0

    for index, entry in enumerate(data):
        messages = entry["messages"]

        if len(messages) == 3:
            expected_roles = ["system", "user", "assistant"]
            category = "single_arithmetic"
        elif len(messages) == 5:
            expected_roles = [
                "system",
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            category = "numeric_update"
        else:
            raise AssertionError(
                f"{path.name}[{index}] 消息数量异常"
            )

        assert [message["role"] for message in messages] == expected_roles

        for message in messages:
            content = message["content"]
            assert isinstance(content, str) and content.strip()
            assert "<|im_start|>" not in content
            assert "<|im_end|>" not in content
            expressions.update(extract_expressions(content))

            if message["role"] != "assistant":
                continue

            assert content.count("<think>") == 1
            assert content.count("</think>") == 1

            reasoning = content.split("<think>")[1].split("</think>")[0]
            final_answer = content.split("</think>")[1].strip()

            assert final_answer.isdigit(), (
                f"{path.name}[{index}] 最终答案不是纯数字：{final_answer}"
            )

            equation_count += check_reasoning(
                reasoning,
                path,
                index,
            )

        categories[category] += 1
        normalized.append(
            json.dumps(messages, ensure_ascii=False, sort_keys=True)
        )

    assert categories == {
        "single_arithmetic": expected_total // 2,
        "numeric_update": expected_total // 2,
    }, f"{path.name} 类别比例异常：{dict(categories)}"

    assert len(set(normalized)) == expected_total, (
        f"{path.name} 内存在重复对话"
    )

    dataset = QwenMultiTurnDataset(data, tokenizer)
    for inputs, targets in dataset:
        assert len(inputs) == len(targets)
        supervised = sum(target != -100 for target in targets)
        assert supervised > 0
        supervised_counts.append(supervised)

    print(f"\n{path.name}")
    print(f"total: {len(data)}, categories: {dict(categories)}")
    print(f"equations checked: {equation_count}")
    print(
        "supervised tokens: "
        f"min={min(supervised_counts)}, "
        f"max={max(supervised_counts)}"
    )
    print("template encoding: passed")

    return data, set(normalized), expressions, equation_count


def collect_existing_expressions():
    expressions = set()

    for name in (
        "i4_multiturn_synthetic_train.json",
        "i4_multiturn_synthetic_val.json",
        "i4_multiturn_smoke_train.json",
        "i4_multiturn_smoke_val.json",
    ):
        data = json.loads(
            (ROOT / "data/instruct" / name).read_text(encoding="utf-8")
        )
        for entry in data:
            for message in entry["messages"]:
                expressions.update(
                    extract_expressions(message["content"])
                )

    return expressions


def check_exclusions(train, val):
    for split_name, data in (("train", train), ("val", val)):
        for index, entry in enumerate(data):
            text = json.dumps(entry, ensure_ascii=False)
            compact = "".join(text.split())

            for literal in (
                "小吕",
                "蓝色",
                "25*24",
                "123+789",
                "17+5",
                "13*5",
                "9+6",
            ):
                assert literal not in compact, (
                    f"{split_name}[{index}] 命中排除项：{literal}"
                )

            assert not ("阿哲" in text and "绿色" in text)


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    train, train_normalized, train_expressions, train_equations = audit_file(
        ROOT / "data/instruct/i4_cot_synthetic_train.json",
        expected_total=128,
        tokenizer=tokenizer,
    )
    val, val_normalized, val_expressions, val_equations = audit_file(
        ROOT / "data/instruct/i4_cot_synthetic_val.json",
        expected_total=32,
        tokenizer=tokenizer,
    )

    assert not train_normalized & val_normalized, (
        "train / val 存在完全重复对话"
    )
    assert not train_expressions & val_expressions, (
        "train / val 存在重复显式算式"
    )

    existing_expressions = collect_existing_expressions()
    cot_expressions = train_expressions | val_expressions
    assert not cot_expressions & existing_expressions, (
        "CoT 数据与 smoke / 非 CoT 数据存在重复显式算式"
    )

    check_exclusions(train, val)

    print("\ntrain / val exact overlap: 0")
    print("train / val expression overlap: 0")
    print("existing-data expression overlap: 0")
    print("exclusion checks: passed")
    print(
        f"total equations checked: {train_equations + val_equations}"
    )
    print("全部 CoT 数据审计通过")


if __name__ == "__main__":
    main()
