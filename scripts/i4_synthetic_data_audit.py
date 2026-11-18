import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from squid.data.chat_sft import QwenMultiTurnDataset


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"


def infer_category(messages):
    if len(messages) == 3:
        return "single_arithmetic"

    first_user = messages[1]["content"]

    if any(
        text in first_user
        for text in ("名字", "称呼", "我叫", "叫我", "昵称")
    ):
        return "name_memory"

    if any(
        text in first_user
        for text in (
            "喜欢",
            "偏爱",
            "最爱",
            "首选",
            "爱吃",
            "爱好",
            "偏好",
            "钟爱",
        )
    ):
        return "preference_memory"

    return "numeric_update"


def audit_file(path, expected_total, tokenizer):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == expected_total, (
        f"{path.name} 数量错误：{len(data)}"
    )

    normalized_conversations = []
    categories = Counter()

    for index, entry in enumerate(data):
        messages = entry["messages"]
        roles = [message["role"] for message in messages]

        if len(messages) == 3:
            assert roles == ["system", "user", "assistant"]
        elif len(messages) == 5:
            assert roles == [
                "system",
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            assert messages[2]["content"] != messages[4]["content"], (
                f"{path.name}[{index}] 后一轮复制了前一轮 Assistant"
            )
        else:
            raise AssertionError(
                f"{path.name}[{index}] 消息数量异常：{len(messages)}"
            )

        for message in messages:
            assert isinstance(message["content"], str)
            assert message["content"].strip()
            assert "<|im_start|>" not in message["content"]
            assert "<|im_end|>" not in message["content"]
            assert "<think>" not in message["content"]

        category = infer_category(messages)

        if (
                len(messages) == 5
                and category == "numeric_update"
                and not any(
            char.isdigit()
            for char in messages[1]["content"]
        )
        ):
            print(
                "待检查样本：",
                index,
                messages[1]["content"],
            )

        categories[category] += 1

        normalized_conversations.append(
            json.dumps(
                messages,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    assert len(set(normalized_conversations)) == expected_total, (
        f"{path.name} 内存在完全重复的对话"
    )

    dataset = QwenMultiTurnDataset(data, tokenizer)
    supervised_counts = []

    for inputs, targets in dataset:
        assert len(inputs) == len(targets)
        supervised = sum(target != -100 for target in targets)
        assert supervised > 0
        supervised_counts.append(supervised)

    expected_by_category = {
        "single_arithmetic": expected_total // 4,
        "name_memory": expected_total // 4,
        "preference_memory": expected_total // 4,
        "numeric_update": expected_total // 4,
    }
    assert dict(categories) == expected_by_category, (
        f"{path.name} 类别数量异常：{dict(categories)}"
    )

    print(f"\n{path.name}")
    print(f"total: {len(data)}")
    print(f"categories: {dict(categories)}")
    print(
        "supervised tokens: "
        f"min={min(supervised_counts)}, "
        f"max={max(supervised_counts)}"
    )
    print("template encoding: passed")

    return data, set(normalized_conversations)


def check_exclusions(train, val):
    forbidden_literals = [
        "小吕",
        "蓝色",
        "25*24",
        "123+789",
        "17+5",
        "13*5",
        "9+6",
    ]

    for split_name, data in (("train", train), ("val", val)):
        for index, entry in enumerate(data):
            text = json.dumps(entry, ensure_ascii=False)
            compact = "".join(text.split())

            for literal in forbidden_literals:
                assert literal not in compact, (
                    f"{split_name}[{index}] 命中排除项：{literal}"
                )

            assert not ("阿哲" in text and "绿色" in text), (
                f"{split_name}[{index}] 复用了阿哲与绿色组合"
            )


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    train, train_normalized = audit_file(
        ROOT / "data/instruct/i4_multiturn_synthetic_train.json",
        expected_total=128,
        tokenizer=tokenizer,
    )
    val, val_normalized = audit_file(
        ROOT / "data/instruct/i4_multiturn_synthetic_val.json",
        expected_total=32,
        tokenizer=tokenizer,
    )

    overlap = train_normalized & val_normalized
    assert not overlap, (
        f"train / val 存在 {len(overlap)} 条完全重复对话"
    )

    check_exclusions(train, val)
    print("\ntrain / val exact overlap: 0")
    print("exclusion checks: passed")
    print("全部数据审计通过")


if __name__ == "__main__":
    main()
