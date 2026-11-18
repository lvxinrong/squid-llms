import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from squid.eval.judge import query_judge as deepseek_query_judge
from squid.eval.pairwise import (
    judge_pair_both_orders,
    reconcile_position_orders,
)
from squid.eval.qwen_judge import query_judge as qwen_query_judge


ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "i5-pairwise-prompt-v1"
QUERY_FUNCTIONS = {
    "deepseek": deepseek_query_judge,
    "qwen": qwen_query_judge,
}
DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "qwen": "qwen4-max",
}
QUERY_SETTINGS = {
    "deepseek": {"temperature": 0, "max_tokens": 16},
    "qwen": {"temperature": 0, "max_tokens": 256},
}


def write_report(path, report):
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def annotate_result(result, pair):
    comparison = result["comparison"]
    consensus = comparison["consensus_winner"]
    counts = pair["character_counts"]
    expected = pair["expected_winner"]

    if counts["A"] == counts["B"]:
        longer_side = None
    else:
        longer_side = "A" if counts["A"] > counts["B"] else "B"

    if consensus is None:
        consensus_role = None
        length_choice = "unresolved"
    elif consensus == "tie":
        consensus_role = "tie"
        length_choice = "tie"
    else:
        consensus_role = pair["roles"][consensus]
        if longer_side is None:
            length_choice = "same_length"
        elif consensus == longer_side:
            length_choice = "longer"
        else:
            length_choice = "shorter"

    result.update({
        "condition": pair["condition"],
        "roles": pair["roles"],
        "character_counts": counts,
        "expected_winner": expected,
        "consensus_role": consensus_role,
        "length_choice": length_choice,
        "expected_agreement": (
            consensus == expected
            if expected is not None and consensus is not None
            else None
        ),
    })
    return result


def build_summary(results):
    comparable = [
        result
        for result in results
        if result["expected_winner"] is not None
        and result["comparison"]["consensus_winner"] is not None
    ]

    return {
        "total": len(results),
        "consistent": sum(
            result["comparison"]["consistent"]
            for result in results
        ),
        "position_sensitive": sum(
            result["comparison"]["position_sensitive"]
            for result in results
        ),
        "decisive_flip": sum(
            result["comparison"]["decisive_flip"]
            for result in results
        ),
        "consensus_roles": dict(Counter(
            result["consensus_role"] or "unresolved"
            for result in results
        )),
        "length_choices": dict(Counter(
            result["length_choice"]
            for result in results
        )),
        "expected_comparable": len(comparable),
        "expected_agreement": sum(
            result["expected_agreement"] is True
            for result in comparable
        ),
        "expected_disagreement": sum(
            result["expected_agreement"] is False
            for result in comparable
        ),
        "expected_unresolved": sum(
            result["expected_winner"] is not None
            and result["comparison"]["consensus_winner"] is None
            for result in results
        ),
        "unprespecified": sum(
            result["expected_winner"] is None
            for result in results
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=tuple(QUERY_FUNCTIONS),
        required=True,
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--fixture",
        default="data/eval/i5-length-control.json",
    )
    args = parser.parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]
    model_slug = model.replace("/", "-")
    output = args.output or (
        f"data/reports/i5-length-{args.provider}-{model_slug}.json"
    )

    fixture_raw = (ROOT / args.fixture).read_bytes()
    fixture = json.loads(fixture_raw)
    assert fixture["status"] == "ready"

    all_pairs = fixture["pairs"]
    selected_pairs = (
        all_pairs[:args.limit]
        if args.limit is not None
        else all_pairs
    )

    identity = {
        "schema_version": "i5-length-judge-v1",
        "prompt_version": PROMPT_VERSION,
        "provider": args.provider,
        "judge_model": model,
        "fixture_sha256": hashlib.sha256(fixture_raw).hexdigest(),
        "query_settings": QUERY_SETTINGS[args.provider],
    }
    output_path = ROOT / output

    if output_path.exists():
        report = json.loads(
            output_path.read_text(encoding="utf-8")
        )
        for key, value in identity.items():
            assert report.get(key) == value, (
                f"续跑身份不一致：{key}"
            )
        report.pop("summary", None)
    else:
        report = {
            **identity,
            "status": "running",
            "pairs": [],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_by_id = {
        pair["pair_id"]: pair
        for pair in report.get("pairs", [])
    }

    def persist(status):
        report["status"] = status
        report["pairs"] = [
            saved_by_id[pair["pair_id"]]
            for pair in all_pairs
            if pair["pair_id"] in saved_by_id
        ]
        write_report(output_path, report)

    api_calls = 0

    def counting_query(prompt, model):
        nonlocal api_calls
        api_calls += 1
        return QUERY_FUNCTIONS[args.provider](
            prompt,
            model=model,
        )

    for pair in selected_pairs:
        pair_id = pair["pair_id"]

        def after_order(orders, pair_id=pair_id):
            saved_by_id[pair_id] = {
                "pair_id": pair_id,
                "judge_model": model,
                "orders": orders,
                "comparison": None,
            }
            persist("running")

        result = judge_pair_both_orders(
            pair,
            counting_query,
            model=model,
            existing=saved_by_id.get(pair_id),
            after_order=after_order,
        )
        saved_by_id[pair_id] = annotate_result(result, pair)
        persist("running")

    selected_ids = {
        pair["pair_id"]
        for pair in selected_pairs
    }
    completed = [
        saved_by_id[pair_id]
        for pair_id in selected_ids
        if saved_by_id[pair_id].get("comparison") is not None
    ]

    report["summary"] = build_summary(completed)
    report["boundary"] = (
        "四个受控合成条件的描述性结果，不构成普遍长度偏见率。"
        "有效信息变长与无效填充变长必须分开解释。"
    )
    persist(
        "complete"
        if len(selected_pairs) == len(all_pairs)
        else "partial"
    )

    for result in completed:
        print(
            f"{result['pair_id']}: "
            f"consensus={result['comparison']['consensus_winner']}, "
            f"role={result['consensus_role']}, "
            f"length={result['length_choice']}, "
            f"expected_agreement={result['expected_agreement']}"
        )
    print(json.dumps(
        report["summary"],
        ensure_ascii=False,
        indent=2,
    ))
    print(f"api_calls_this_run: {api_calls}")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
