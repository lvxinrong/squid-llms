import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from squid.data.chat_sft import encode_qwen_multiturn


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"
SOURCE = ROOT / "data/instruct/i6_l0_concept_train_v1.json"
EVAL_SOURCE = ROOT / "data/instruct/i6_l0_concept_eval_v1.json"
OUTPUT_DIR = ROOT / "data/instruct"
TOPIC_PREFIXES = (
    "L0-STATIC",
    "L0-FINAL",
    "L0-DIV",
    "L0-EQ",
)


def build_entry(task, data_role):
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名严谨的 Java 教师。请直接回答问题，"
                "先给结论，再用简洁的话解释原因。"
                "不要重复题目，不要输出与问题无关的内容。"
            ),
        },
        {
            "role": "user",
            "content": task["question"],
        },
        {
            "role": "assistant",
            "content": task["reference_answer"],
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    full_ids, targets = encode_qwen_multiturn(
        tokenizer,
        messages,
        enable_thinking=False,
    )

    return {
        "instance_id": task["id"],
        "data_role": data_role,
        "messages": messages,
        "scoring_points": task["scoring_points"],
        "notes": task["notes"],
        "token_audit": {
            "full_tokens": len(full_ids),
            "supervised_tokens": sum(
                target != -100
                for target in targets
            ),
        },
    }


def split_rows(rows):
    train_rows = []
    validation_rows = []

    for prefix in TOPIC_PREFIXES:
        group = [
            row
            for row in rows
            if row["id"].startswith(prefix)
        ]
        assert len(group) == 25, (prefix, len(group))
        train_rows.extend(group[:-2])
        validation_rows.extend(group[-2:])

    return train_rows, validation_rows


def write_entries(path, entries):
    ids = [entry["instance_id"] for entry in entries]
    assert len(ids) == len(set(ids)), path

    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    token_counts = [
        entry["token_audit"]["full_tokens"]
        for entry in entries
    ]
    supervised_counts = [
        entry["token_audit"]["supervised_tokens"]
        for entry in entries
    ]
    assert max(token_counts) <= 4096, path

    print(
        path,
        "samples=", len(entries),
        "max_full_tokens=", max(token_counts),
        "max_supervised_tokens=", max(supervised_counts),
    )


def main():
    source_rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert len(source_rows) == 100
    assert all(row["data_role"] == "train" for row in source_rows)

    train_rows, validation_rows = split_rows(source_rows)
    assert len(train_rows) == 92
    assert len(validation_rows) == 8

    train_entries = [
        build_entry(row, "train")
        for row in train_rows
    ]
    validation_entries = [
        build_entry(row, "validation")
        for row in validation_rows
    ]

    write_entries(
        OUTPUT_DIR / "i6_l0_concept_sft_train.json",
        train_entries,
    )
    write_entries(
        OUTPUT_DIR / "i6_l0_concept_sft_val.json",
        validation_entries,
    )

    eval_rows = json.loads(EVAL_SOURCE.read_text(encoding="utf-8"))
    assert len(eval_rows) == 20
    assert all(
        row["data_role"] == "eval_holdout"
        for row in eval_rows
    )

    holdout_entries = [
        build_entry(row, "eval_holdout")
        for row in eval_rows
    ]
    write_entries(
        OUTPUT_DIR / "i6_l0_concept_sft_holdout.json",
        holdout_entries,
    )

    all_ids = {
        entry["instance_id"]
        for entry in train_entries + validation_entries + holdout_entries
    }
    assert len(all_ids) == 120, len(all_ids)
    print("all ids unique across splits: passed")
    print("roles:", dict(Counter(
        entry["data_role"]
        for entry in train_entries + validation_entries + holdout_entries
    )))


if __name__ == "__main__":
    main()
