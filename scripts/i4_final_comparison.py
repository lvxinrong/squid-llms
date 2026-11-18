import itertools
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REPORTS = {
    "original": "i4-cot-original-baseline.json",
    "cot_sft_v2": "i4-cot-sft-v2-lr1e-6-behavior.json",
    "dpo_v1": "i4-dpo-v1-cot-behavior.json",
    "grpo_one_group": "i4-grpo-one-group-cot-behavior.json",
}


def load_report(filename):
    report = json.loads(
        (ROOT / "data/reports" / filename).read_text(encoding="utf-8")
    )
    cases = sorted(report["cases"], key=lambda case: case["index"])
    tokens = [case["generated_tokens"] for case in cases]

    return report, {
        "total": len(cases),
        "answer_correct": sum(
            case["answer_correct"] for case in cases
        ),
        "format_compliant": sum(
            case["format_compliant"] for case in cases
        ),
        "passed": sum(case["passed"] for case in cases),
        "natural_stop": sum(
            case["finish_reason"] == "stop" for case in cases
        ),
        "length_limited": sum(
            case["finish_reason"] == "length" for case in cases
        ),
        "mean_generated_tokens": statistics.mean(tokens),
        "median_generated_tokens": statistics.median(tokens),
        "max_generated_tokens": max(tokens),
    }


def compare(left_cases, right_cases):
    wins = losses = ties = 0

    for left, right in zip(left_cases, right_cases):
        assert left["index"] == right["index"]

        if left["passed"] and not right["passed"]:
            wins += 1
        elif right["passed"] and not left["passed"]:
            losses += 1
        else:
            ties += 1

    decisive = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_all": wins / (wins + losses + ties),
        "decisive_win_rate": (
            wins / decisive if decisive else None
        ),
    }


def main():
    loaded = {
        name: load_report(filename)
        for name, filename in REPORTS.items()
    }

    summaries = {
        name: summary
        for name, (_, summary) in loaded.items()
    }

    pairwise = {}
    for left_name, right_name in itertools.combinations(REPORTS, 2):
        left_cases = loaded[left_name][0]["cases"]
        right_cases = loaded[right_name][0]["cases"]

        pairwise[f"{left_name} vs {right_name}"] = compare(
            left_cases,
            right_cases,
        )

    report = {
        "models": summaries,
        "pairwise": pairwise,
        "definitions": {
            "win": "同一题上，左侧 passed=true 且右侧 passed=false",
            "loss": "同一题上，右侧 passed=true 且左侧 passed=false",
            "win_rate_all": "wins / 全部题目数，平局计入分母",
            "decisive_win_rate": "wins / (wins + losses)，不计平局",
        },
        "boundary": (
            "这是同一 32 条 CoT 样本上的规则评委对照，"
            "不是经过 I5 校准的 LLM 评委胜率，也不代表开放域能力。"
        ),
    }

    output = ROOT / "data/reports/i4-final-comparison.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(json.dumps(pairwise, ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
