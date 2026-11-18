import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
USER = "21 乘以 17 是多少？简短思考后，最终只输出一个数字。"
EXPECTED_ANSWER = "357"

BASE_RESPONSE = """<think>
把 21 拆成 20 和 1。
20 × 17 = 340，1 × 17 = 17。
340 + 17 = 357。
</think>

357"""

PADDED_RESPONSE = BASE_RESPONSE.replace(
    "340 + 17 = 357。",
    """340 + 17 = 357。
用户可能希望得到简短回答，因此需要关注最终格式。
还需要再次确认最终只输出一个数字，不添加额外解释。
计算已经给出结果，最终结果仍然是前面计算得到的数字。""",
)

INFORMATIVE_RESPONSE = BASE_RESPONSE.replace(
    "\n</think>",
    "\n换一种拆分验算：21 × (10 + 7) = 210 + 147 = 357。\n</think>",
)

WRONG_VERBOSE_RESPONSE = """<think>
把 21 拆成 20 和 1。
20 × 17 = 340，1 × 17 = 17。
把两部分相加，得到 347。
用户可能希望得到简短回答，因此需要关注最终格式。
还需要再次确认最终只输出一个数字，不添加额外解释。
计算已经给出结果，最终直接输出前面得到的数字。
</think>

347"""


def make_pair(
    pair_id,
    condition,
    response_a,
    response_b,
    role_a,
    role_b,
    expected_winner,
):
    return {
        "pair_id": pair_id,
        "condition": condition,
        "history": [],
        "user": USER,
        "expected_answer": EXPECTED_ANSWER,
        "responses": {
            "A": response_a,
            "B": response_b,
        },
        "roles": {
            "A": role_a,
            "B": role_b,
        },
        "expected_winner": expected_winner,
        "character_counts": {
            "A": len(response_a),
            "B": len(response_b),
        },
    }


def build_pairs():
    return [
        make_pair(
            "length-01-exact",
            "exact_duplicate",
            BASE_RESPONSE,
            BASE_RESPONSE,
            "identical",
            "identical",
            "tie",
        ),
        make_pair(
            "length-02-padding",
            "padded_verbosity",
            BASE_RESPONSE,
            PADDED_RESPONSE,
            "concise",
            "padded",
            "A",
        ),
        make_pair(
            "length-03-informative",
            "informative_length",
            BASE_RESPONSE,
            INFORMATIVE_RESPONSE,
            "concise",
            "informative",
            None,
        ),
        make_pair(
            "length-04-correctness",
            "correctness_gate",
            BASE_RESPONSE,
            WRONG_VERBOSE_RESPONSE,
            "correct_concise",
            "wrong_verbose",
            "A",
        ),
    ]


def main():
    document = {
        "schema_version": "i5-length-control-v1",
        "status": "ready",
        "pairs": build_pairs(),
        "boundary": (
            "字符数只用于确认长度操纵方向，不代表 token 数。"
            "informative_length 不预设人工赢家。"
        ),
    }

    output_path = ROOT / "data/eval/i5-length-control.json"
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for pair in document["pairs"]:
        counts = pair["character_counts"]
        print(
            f"{pair['pair_id']}: "
            f"A={counts['A']} chars, "
            f"B={counts['B']} chars, "
            f"expected={pair['expected_winner']}"
        )
    print(f"fixture: {output_path}")


if __name__ == "__main__":
    main()
