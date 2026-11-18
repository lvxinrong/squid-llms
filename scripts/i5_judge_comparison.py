import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDERS = ("AB", "BA")
RELATIONS = (
    "agree",
    "disagree",
    "deepseek_only_unresolved",
    "qwen_only_unresolved",
    "both_unresolved",
)

def consensus_relation(deepseek_winner, qwen_winner):
    if deepseek_winner is None and qwen_winner is None:
        return "both_unresolved"
    if deepseek_winner is None:
        return "deepseek_only_unresolved"
    if qwen_winner is None:
        return "qwen_only_unresolved"
    if deepseek_winner == qwen_winner:
        return "agree"
    return "disagree"

def judge_view(result, human_winner):
    assert result["human_winner"] == human_winner

    comparison = result["comparison"]
    consensus = comparison["consensus_winner"]

    return {
        "consensus_winner": consensus,
        "consistent": comparison["consistent"],
        "position_sensitive": comparison["position_sensitive"],
        "decisive_flip": comparison["decisive_flip"],
        "human_agreement": (
            consensus == human_winner
            if consensus is not None
            else None
        ),
        "orders": {
            order: result["orders"][order]["winner"]
            for order in ORDERS
        },
    }

def summarize_judge(pairs, judge_name):
    views = [
        pair["judges"][judge_name]
        for pair in pairs
    ]
    winners = Counter(
        view["consensus_winner"] or "unresolved"
        for view in views
    )

    return {
        "total": len(views),
        "consistent": sum(
            view["consistent"]
            for view in views
        ),
        "position_sensitive": sum(
            view["position_sensitive"]
            for view in views
        ),
        "decisive_flip": sum(
            view["decisive_flip"]
            for view in views
        ),
        "consensus_available": sum(
            view["consensus_winner"] is not None
            for view in views
        ),
        "consensus_winners": dict(winners),
        "human_agreement": sum(
            view["human_agreement"] is True
            for view in views
        ),
        "human_disagreement": sum(
            view["human_agreement"] is False
            for view in views
        ),
        "human_unresolved": sum(
            view["human_agreement"] is None
            for view in views
        ),
    }

def build_comparison(
    blind,
    deepseek_report,
    qwen_report,
    blind_sha256,
):
    assert blind["status"] == "labeled"
    reports = {
        "deepseek": deepseek_report,
        "qwen": qwen_report,
    }
    prompt_version = deepseek_report["prompt_version"]
    blind_by_id = {
        pair["pair_id"]: pair
        for pair in blind["pairs"]
    }

    loaded_by_judge = {}
    metadata = {}

    for name, report in reports.items():
        assert report["blind_sha256"] == blind_sha256
        assert report["prompt_version"] == prompt_version

        by_id = {
            pair["pair_id"]: pair
            for pair in report["pairs"]
        }
        assert set(by_id) == set(blind_by_id)
        assert all(
            result.get("comparison") is not None
            for result in by_id.values()
        )
        assert all(
            result["orders"][order]["winner"] in {"A", "B", "tie"}
            for result in by_id.values()
            for order in ORDERS
        )

        loaded_by_judge[name] = by_id
        metadata[name] = {
            "provider": report.get("provider"),
            "model": report["judge_model"],
            "query_settings": report["query_settings"],
            "last_run_status": report["status"],
        }


    pairs = []
    relation_counts = {
        relation: 0
        for relation in RELATIONS
    }
    raw_order_agreement = 0
    raw_order_disagreements = []

    for pair_id in blind_by_id:
        human = blind_by_id[pair_id]["human"]
        views = {
            name: judge_view(
            loaded_by_judge[name][pair_id],
            human["winner"],
            )
            for name in reports
        }

        relation = consensus_relation(
            views["deepseek"]["consensus_winner"],
            views["qwen"]["consensus_winner"],
        )
        relation_counts[relation] += 1

        for order in ORDERS:
            deepseek_winner = views["deepseek"]["orders"][order]
            qwen_winner = views["qwen"]["orders"][order]

            if deepseek_winner == qwen_winner:
                raw_order_agreement += 1
            else:
                raw_order_disagreements.append({
                    "pair_id": pair_id,
                    "order": order,
                    "deepseek": deepseek_winner,
                    "qwen": qwen_winner,
                })

        pairs.append({
            "pair_id": pair_id,
            "human_winner": human["winner"],
            "human_reason_codes": human["reason_codes"],
            "judges": views,
            "consensus_relation": relation,
        })

    comparable = (
        relation_counts["agree"]
        + relation_counts["disagree"]
    )

    return {
        "schema_version": "i5-judge-comparison-v1",
        "metadata": {
            "blind_sha256": blind_sha256,
            "prompt_version": prompt_version,
            "judges": metadata,
        },
        "summary": {
            "total_pairs": len(pairs),
            "human_winners": dict(Counter(
                pair["human_winner"]
                for pair in pairs
            )),
            "judges": {
                name: summarize_judge(pairs, name)
                for name in reports
            },
            "cross_judge": {
                **relation_counts,
                "consensus_comparable": comparable,
                "raw_order_comparable": len(pairs) * len(ORDERS),
                "raw_order_agreement": raw_order_agreement,
                "raw_order_disagreement": len(
                    raw_order_disagreements
                ),
                "raw_order_disagreements": (
                    raw_order_disagreements
                ),
            },
        },
        "pairs": pairs,
        "boundary": (
            "这是同一批少量分层试点的人工与双评委描述性对照，"
            "不用于普遍准确率、统计相关性或模型排名。"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blind",
        default="data/eval/i5-human-calibration-blind.json",
    )
    parser.add_argument(
        "--deepseek",
        default="data/reports/i5-pairwise-deepseek-chat-pilot.json",
    )
    parser.add_argument(
        "--qwen",
        default="data/reports/i5-pairwise-qwen4-max-pilot.json",
    )
    parser.add_argument(
        "--output",
        default="data/reports/i5-judge-comparison-pilot.json",
    )

    args = parser.parse_args()

    blind_raw = (ROOT / args.blind).read_bytes()
    blind = json.loads(blind_raw)
    deepseek = json.loads((ROOT / args.deepseek).read_text(encoding="utf-8"))
    qwen = json.loads((ROOT / args.qwen).read_text(encoding="utf-8"))

    report = build_comparison(
        blind,
        deepseek,
        qwen,
        hashlib.sha256(blind_raw).hexdigest(),
    )

    output_path = ROOT / args.output
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        report["summary"],
        ensure_ascii=False,
        indent=2,
    ))
    print(f"report: {output_path}")


if __name__ == '__main__':
    main()
