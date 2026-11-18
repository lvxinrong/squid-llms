import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261118

REPORTS = {
    "original": "i4-cot-original-baseline.json",
    "cot_sft_v2": "i4-cot-sft-v2-lr1e-6-behavior.json",
    "dpo_v1": "i4-dpo-v1-cot-behavior.json",
    "grpo_one_group": "i4-grpo-one-group-cot-behavior.json",
}

TARGETS = {
    "rule_disagreement": 3,
    "both_passed": 3,
    "both_failed": 2,
}

REASON_CODES = [
    "correctness",
    "instruction",
    "reasoning",
    "clarity",
]


def classify_pair(left, right):
    if left["passed"] != right["passed"]:
        return "rule_disagreement"
    if left["passed"]:
        return "both_passed"
    return "both_failed"


def task_signature(case):
    return (
        case["category"],
        case["history"],
        case["user"],
        case["expected_answer"],
    )


def collect_candidates(cases_by_model):
    index_sets = [
        set(cases)
        for cases in cases_by_model.values()
    ]
    assert index_sets
    assert all(indices == index_sets[0] for indices in index_sets)

    candidates = []

    for index in sorted(index_sets[0]):
        for left_name, right_name in itertools.combinations(cases_by_model, 2):
            left = cases_by_model[left_name][index]
            right = cases_by_model[right_name][index]

            assert task_signature(left) == task_signature(right)

            candidates.append({
                "case_index": index,
                "category": left["category"],
                "history": left["history"],
                "user": left["user"],
                "expected_answer": left["expected_answer"],
                "stratum": classify_pair(left, right),
                "left_model": left_name,
                "right_model": right_name,
                "left": left,
                "right": right,
            })

    return candidates


def select_candidates(candidates, targets=TARGETS, seed=SEED):
    rng = random.Random(seed)
    selected = []
    used_cases = set()

    for stratum, target in targets.items():
        group = [
            candidate
            for candidate in candidates
            if candidate["stratum"] == stratum
        ]
        rng.shuffle(group)

        chosen = []
        for candidate in group:
            case_index = candidate["case_index"]
            if case_index in used_cases:
                continue

            chosen.append(candidate)
            used_cases.add(case_index)

            if len(chosen) == target:
                break

        assert len(chosen) == target, (
            f"{stratum} 按 case_index 去重后只有 "
            f"{len(chosen)} 对，无法抽取 {target} 对"
        )
        selected.extend(chosen)

    rng.shuffle(selected)
    return selected



def build_blind_pair(candidate, pair_id, rng):
    sides = [
        (candidate["left_model"], candidate["left"]),
        (candidate["right_model"], candidate["right"]),
    ]
    rng.shuffle(sides)

    public = {
        "pair_id": pair_id,
        "category": candidate["category"],
        "history": candidate["history"],
        "user": candidate["user"],
        "expected_answer": candidate["expected_answer"],
        "responses": {},
        "human": {
            "winner": None,
            "reason_codes": [],
            "notes": "",
        },
    }

    private = {
        "pair_id": pair_id,
        "case_index": candidate["case_index"],
        "stratum": candidate["stratum"],
        "sides": {},
    }

    for label, (model_name, case) in zip(("A", "B"), sides):
        public["responses"][label] = case["response"]
        private["sides"][label] = {
            "model": model_name,
            "passed": case["passed"],
            "generated_tokens": case["generated_tokens"],
        }

    return public, private

def assert_safe_to_export(blind_path):
    if not blind_path.exists():
        return

    document = json.loads(
        blind_path.read_text(encoding="utf-8")
    )
    has_labels = (
        document.get("status") == "labeled"
        or any(
            pair.get("human", {}).get("winner") is not None
            for pair in document.get("pairs", [])
        )
    )

    if has_labels:
        raise RuntimeError(
            f"检测到人工标签，拒绝覆盖：{blind_path}。"
            "如需创建新批次，请修改输出文件名。"
        )

def main():
    cases_by_model = {}
    sources = {}

    for model_name, filename in REPORTS.items():
        path = ROOT / "data/reports" / filename
        raw = path.read_bytes()
        report = json.loads(raw)

        cases_by_model[model_name] = {
            case["index"]: case
            for case in report["cases"]
        }

        sources[model_name] = {
            "path": f"data/reports/{filename}",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    candidates = collect_candidates(cases_by_model)
    selected = select_candidates(candidates)

    public_pairs = []
    private_pairs = []
    position_rng = random.Random(SEED + 1)

    for number, candidate in enumerate(selected, 1):
        pair_id = f"i5-pilot-{number:02d}"
        public, private = build_blind_pair(
            candidate,
            pair_id,
            position_rng,
        )
        public_pairs.append(public)
        private_pairs.append(private)

    blind_document = {
        "schema_version": "i5-human-calibration-v1",
        "status": "unlabeled",
        "winner_values": ["A", "B", "tie"],
        "reason_codes": REASON_CODES,
        "instruction": (
            "只比较当前回答。正确性优先；随后检查指令、推理和有效信息密度。"
            "没有实质差异时选择 tie。不要根据回答长短猜测模型能力。"
        ),
        "pairs": public_pairs,
    }

    key_document = {
        "schema_version": "i5-human-calibration-v1",
        "seed": SEED,
        "sources": sources,
        "selection": dict(TARGETS),
        "pairs": private_pairs,
    }

    output_dir = ROOT / "data/eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    blind_path = output_dir / "i5-human-calibration-blind.json"
    assert_safe_to_export(blind_path)

    key_path = output_dir / "i5-human-calibration-key.json"

    blind_path.write_text(
        json.dumps(blind_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    key_path.write_text(
        json.dumps(key_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"pairs: {len(public_pairs)}")
    print(dict(Counter(
        pair["stratum"]
        for pair in private_pairs
    )))
    print(f"blind: {blind_path}")
    print(f"key: {key_path}（人工标注完成前不要打开）")


if __name__ == "__main__":
    main()
