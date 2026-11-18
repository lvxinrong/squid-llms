import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261119
REASON_CODES = [
    "correctness",
    "instruction",
    "reasoning",
    "clarity",
]
PADDING = (
    "用户可能希望得到简短回答，因此需要关注最终格式。"
    "还需要再次确认最终只输出一个数字，不添加额外解释。"
    "计算已经给出结果，最终直接输出前面得到的数字。"
)


def cot(reasoning, final_answer):
    return f"<think>\n{reasoning}\n</think>\n\n{final_answer}"


def add_reasoning(response, extra):
    return response.replace(
        "\n</think>",
        f"\n{extra}\n</think>",
    )


def candidate(
    case_id,
    condition,
    user,
    expected_answer,
    variants,
    preferred_role,
    history=None,
):
    return {
        "case_id": case_id,
        "condition": condition,
        "history": history or [],
        "user": user,
        "expected_answer": expected_answer,
        "variants": variants,
        "preferred_role": preferred_role,
    }


def build_candidates():
    correct_36 = cot(
        "36 × 12 = 36 × (10 + 2) = 360 + 72 = 432。",
        "432",
    )
    wrong_36 = cot(
        "36 × 10 = 360，36 × 2 = 62，合计 422。" + PADDING,
        "422",
    )

    correct_books = cot("60 - 15 = 45。", "45")
    wrong_books = cot(
        "48 + 12 = 70，再减去 15，剩余 55。" + PADDING,
        "55",
    )

    base_47 = cot(
        "40 + 30 = 70，7 + 8 = 15，合计 85。",
        "85",
    )
    informative_47 = add_reasoning(
        base_47,
        "独立验算：47 + 40 = 87，再减去多算的 2，得到 85。",
    )

    base_24 = cot(
        "20 × 5 = 100，4 × 5 = 20，合计 120。",
        "120",
    )
    informative_24 = add_reasoning(
        base_24,
        "独立验算：24 × 10 = 240，240 ÷ 2 = 120。",
    )

    concise_63 = cot(
        "60 + 20 = 80，3 + 9 = 12，合计 92。",
        "92",
    )
    padded_63 = add_reasoning(concise_63, PADDING)

    concise_pens = cot("32 - 9 = 23。", "23")
    padded_pens = add_reasoning(concise_pens, PADDING)

    return [
        candidate(
            "holdout-correctness-01",
            "correctness_gate",
            "36 乘以 12 是多少？简短思考后，最终只输出数字。",
            "432",
            [
                {"role": "correct", "response": correct_36},
                {"role": "wrong_verbose", "response": wrong_36},
            ],
            "correct",
        ),
        candidate(
            "holdout-correctness-02",
            "correctness_gate",
            "后来借出 15 本，还剩多少本？最终只输出数字。",
            "45",
            [
                {"role": "correct", "response": correct_books},
                {"role": "wrong_verbose", "response": wrong_books},
            ],
            "correct",
            history=[[
                "书架原有 48 本书，又放进 12 本，现在有多少本？",
                cot("48 + 12 = 60。", "60"),
            ]],
        ),
        candidate(
            "holdout-informative-01",
            "informative_length",
            "47 + 38 等于多少？简短思考后，最终只输出数字。",
            "85",
            [
                {"role": "concise", "response": base_47},
                {"role": "informative", "response": informative_47},
            ],
            "informative",
        ),
        candidate(
            "holdout-informative-02",
            "informative_length",
            "24 乘以 5 是多少？简短思考后，最终只输出数字。",
            "120",
            [
                {"role": "concise", "response": base_24},
                {"role": "informative", "response": informative_24},
            ],
            "informative",
        ),
        candidate(
            "holdout-padding-01",
            "padded_verbosity",
            "63 + 29 等于多少？简短思考后，最终只输出数字。",
            "92",
            [
                {"role": "concise", "response": concise_63},
                {"role": "padded", "response": padded_63},
            ],
            "concise",
        ),
        candidate(
            "holdout-padding-02",
            "padded_verbosity",
            "再拿走 9 支，现在剩多少支？最终只输出数字。",
            "23",
            [
                {"role": "concise", "response": concise_pens},
                {"role": "padded", "response": padded_pens},
            ],
            "concise",
            history=[[
                "盒子里有 25 支笔，又放入 7 支，现在有多少支？",
                cot("25 + 7 = 32。", "32"),
            ]],
        ),
        candidate(
            "holdout-equivalent-01",
            "equivalent",
            "90 - 37 等于多少？简短思考后，最终只输出数字。",
            "53",
            [
                {
                    "role": "equivalent_1",
                    "response": cot(
                        "90 - 30 = 60，再减 7，得到 53。",
                        "53",
                    ),
                },
                {
                    "role": "equivalent_2",
                    "response": cot(
                        "把 37 拆成 30 和 7；90 - 30 = 60，60 - 7 = 53。",
                        "53",
                    ),
                },
            ],
            None,
        ),
        candidate(
            "holdout-equivalent-02",
            "equivalent",
            "第二天又收集 6 枚，合计多少枚？最终只输出数字。",
            "38",
            [
                {
                    "role": "equivalent_1",
                    "response": cot("32 + 6 = 38。", "38"),
                },
                {
                    "role": "equivalent_2",
                    "response": cot(
                        "已有总数是 32，加上第二天的 6，合计 38。",
                        "38",
                    ),
                },
            ],
            None,
            history=[[
                "上午收集 18 枚邮票，下午收集 14 枚，合计多少枚？",
                cot("18 + 14 = 32。", "32"),
            ]],
        ),
        candidate(
            "holdout-instruction-01",
            "instruction_following",
            "计算 56 + 27，简短思考，最终只输出一个数字，不要带单位。",
            "83",
            [
                {
                    "role": "instruction_compliant",
                    "response": cot("56 + 27 = 83。", "83"),
                },
                {
                    "role": "instruction_violation",
                    "response": cot("56 + 27 = 83。", "83 个"),
                },
            ],
            "instruction_compliant",
        ),
        candidate(
            "holdout-instruction-02",
            "instruction_following",
            "现在搬走 18 个，最终只回答剩余数字，不要解释。",
            "30",
            [
                {
                    "role": "instruction_compliant",
                    "response": cot("48 - 18 = 30。", "30"),
                },
                {
                    "role": "instruction_violation",
                    "response": cot(
                        "48 - 18 = 30。",
                        "还剩 30 个箱子。",
                    ),
                },
            ],
            "instruction_compliant",
            history=[[
                "仓库有 40 个箱子，又补进 8 个，现在有多少个？",
                cot("40 + 8 = 48。", "48"),
            ]],
        ),
    ]


def build_documents(candidates, seed=SEED):
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    public_pairs = []
    private_pairs = []

    for number, item in enumerate(shuffled, 1):
        pair_id = f"i5-holdout-{number:02d}"
        variants = list(item["variants"])
        rng.shuffle(variants)

        responses = {}
        sides = {}

        for label, variant in zip(("A", "B"), variants):
            responses[label] = variant["response"]
            sides[label] = {
                "role": variant["role"],
                "characters": len(variant["response"]),
            }

        if item["preferred_role"] is None:
            hypothesis = "tie"
        else:
            hypothesis = next(
                label
                for label, side in sides.items()
                if side["role"] == item["preferred_role"]
            )

        public_pairs.append({
            "pair_id": pair_id,
            "history": item["history"],
            "user": item["user"],
            "expected_answer": item["expected_answer"],
            "responses": responses,
            "human": {
                "winner": None,
                "reason_codes": [],
                "notes": "",
            },
        })
        private_pairs.append({
            "pair_id": pair_id,
            "case_id": item["case_id"],
            "condition": item["condition"],
            "sides": sides,
            "design_hypothesis": hypothesis,
        })

    blind = {
        "schema_version": "i5-holdout-v1",
        "status": "unlabeled",
        "winner_values": ["A", "B", "tie"],
        "reason_codes": REASON_CODES,
        "instruction": (
            "正确性优先。正确、非重复、相关的独立验算可以构成优势；"
            "无效重复不算增量。没有实质差异时选择 tie。"
        ),
        "pairs": public_pairs,
    }
    key = {
        "schema_version": "i5-holdout-v1",
        "seed": seed,
        "pairs": private_pairs,
    }
    return blind, key


def assert_safe_to_export(blind_path):
    if not blind_path.exists():
        return

    document = json.loads(blind_path.read_text(encoding="utf-8"))
    has_labels = (
        document.get("status") == "labeled"
        or any(
            pair.get("human", {}).get("winner") is not None
            for pair in document.get("pairs", [])
        )
    )
    if has_labels:
        raise RuntimeError("检测到 held-out 人工标签，拒绝覆盖。")


def main():
    blind, key = build_documents(build_candidates())
    output_dir = ROOT / "data/eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    blind_path = output_dir / "i5-holdout-blind.json"
    key_path = output_dir / "i5-holdout-key.json"
    assert_safe_to_export(blind_path)

    blind_path.write_text(
        json.dumps(blind, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    key_path.write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"pairs: {len(blind['pairs'])}")
    print(f"blind: {blind_path}")
    print(f"key: {key_path}（人工标注完成前不要打开）")


if __name__ == "__main__":
    main()
