import json
import runpy
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"

cot_audit = runpy.run_path(
    str(ROOT / "scripts/i4_cot_data_audit.py")
)
cot_eval = runpy.run_path(
    str(ROOT / "scripts/i4_cot_behavior_eval.py")
)
logprob_probe = runpy.run_path(
    str(ROOT / "scripts/i4_dpo_logprob_probe.py")
)

extract_expressions = cot_audit["extract_expressions"]
check_reasoning = cot_audit["check_reasoning"]
EQUATION_RE = cot_audit["EQUATION_RE"]
split_cot_response = cot_eval["split_cot_response"]
has_complete_cot_format = cot_eval["has_complete_cot_format"]
encode_response = logprob_probe["encode_response"]


def equation_errors(text):
    errors = []

    for left, operator, right, result in EQUATION_RE.findall(text):
        left = int(left)
        right = int(right)
        result = int(result)

        if operator in {"×", "*"}:
            expected = left * right
        elif operator == "+":
            expected = left + right
        else:
            expected = left - right

        if expected != result:
            errors.append((left, operator, right, result, expected))

    return errors


def audit_file(path, expected_total, tokenizer):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == expected_total

    pair_types = Counter()
    task_types = Counter()
    normalized_prompts = []
    chosen_equations = 0
    verbose_wrong_with_errors = 0

    for index, entry in enumerate(data):
        pair_type = entry["pair_type"]
        messages = entry["messages"]
        chosen = entry["chosen"]
        rejected = entry["rejected"]

        if len(messages) == 2:
            expected_roles = ["system", "user"]
            task_type = "single_arithmetic"
        elif len(messages) == 4:
            expected_roles = [
                "system",
                "user",
                "assistant",
                "user",
            ]
            task_type = "numeric_update"
        else:
            raise AssertionError(
                f"{path.name}[{index}] 消息数量异常"
            )

        assert [message["role"] for message in messages] == expected_roles
        assert messages[-1]["role"] == "user"

        for response_name, response in (
            ("chosen", chosen),
            ("rejected", rejected),
        ):
            assert "<|im_start|>" not in response
            assert "<|im_end|>" not in response
            encode_response(
                tokenizer,
                messages,
                response,
            )

        chosen_reasoning, chosen_answer = split_cot_response(chosen)
        assert has_complete_cot_format(chosen)
        assert chosen_answer.isdigit()
        chosen_equations += check_reasoning(
            chosen_reasoning,
            path,
            index,
        )

        rejected_reasoning, rejected_answer = split_cot_response(rejected)
        rejected_errors = equation_errors(rejected_reasoning)

        if pair_type == "verbose_correct":
            assert has_complete_cot_format(rejected)
            assert rejected_answer == chosen_answer
            assert not rejected_errors
            assert rejected != chosen
        elif pair_type == "verbose_wrong":
            if rejected_answer != chosen_answer or rejected_errors:
                verbose_wrong_with_errors += 1
        elif pair_type == "repetition":
            assert any(
                text in rejected_reasoning
                for text in ("是不是", "再算", "确认", "不对")
            ), f"{path.name}[{index}] 未体现重复或自我怀疑"
        elif pair_type == "malformed":
            assert not has_complete_cot_format(rejected), (
                f"{path.name}[{index}] malformed 反而结构完整"
            )
        else:
            raise AssertionError(f"未知 pair_type：{pair_type}")

        pair_types[pair_type] += 1
        task_types[task_type] += 1
        normalized_prompts.append(
            json.dumps(messages, ensure_ascii=False, sort_keys=True)
        )

    expected_pair_types = {
        "verbose_correct": expected_total // 4,
        "verbose_wrong": expected_total // 4,
        "repetition": expected_total // 4,
        "malformed": expected_total // 4,
    }
    assert dict(pair_types) == expected_pair_types, (
        f"{path.name} pair_type 数量异常：{dict(pair_types)}"
    )

    assert task_types == {
        "single_arithmetic": expected_total // 2,
        "numeric_update": expected_total // 2,
    }, f"{path.name} 任务类型比例异常：{dict(task_types)}"

    assert len(set(normalized_prompts)) == expected_total, (
        f"{path.name} 内存在重复 Prompt"
    )
    assert verbose_wrong_with_errors == expected_total // 4, (
        f"{path.name} verbose_wrong 未都包含明确错误"
    )

    print(f"\n{path.name}")
    print(f"total: {len(data)}")
    print(f"pair types: {dict(pair_types)}")
    print(f"task types: {dict(task_types)}")
    print(f"chosen equations checked: {chosen_equations}")
    print("preference encoding: passed")

    return data, set(normalized_prompts)


def collect_existing_expressions():
    expressions = set()

    for name in (
        "i4_multiturn_synthetic_train.json",
        "i4_multiturn_synthetic_val.json",
        "i4_multiturn_smoke_train.json",
        "i4_multiturn_smoke_val.json",
        "i4_cot_synthetic_train.json",
        "i4_cot_synthetic_val.json",
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
                "37×6",
                "37*6",
            ):
                assert literal not in compact, (
                    f"{split_name}[{index}] 命中排除项：{literal}"
                )


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    train, train_prompts = audit_file(
        ROOT / "data/instruct/i4_dpo_synthetic_train.json",
        expected_total=32,
        tokenizer=tokenizer,
    )
    val, val_prompts = audit_file(
        ROOT / "data/instruct/i4_dpo_synthetic_val.json",
        expected_total=8,
        tokenizer=tokenizer,
    )

    assert not train_prompts & val_prompts, (
        "train / val 存在重复 Prompt"
    )

    dpo_expressions = set()
    for entry in train + val:
        for message in entry["messages"]:
            dpo_expressions.update(
                extract_expressions(message["content"])
            )
        dpo_expressions.update(
            extract_expressions(entry["chosen"])
        )
        dpo_expressions.update(
            extract_expressions(entry["rejected"])
        )

    existing_expressions = collect_existing_expressions()
    assert not dpo_expressions & existing_expressions, (
        "DPO 数据与已有数据存在重复显式算式"
    )

    check_exclusions(train, val)

    print("\ntrain / val prompt overlap: 0")
    print("existing-data expression overlap: 0")
    print("exclusion checks: passed")
    print("全部 DPO 数据审计通过")


if __name__ == "__main__":
    main()
