import argparse
import hashlib
import json
from pathlib import Path

from squid.eval.judge import query_judge as deepseek_query_judge
from squid.eval.pairwise import (
    PAIRWISE_PROMPT_V2,
    build_pairwise_prompt,
    build_pairwise_prompt_v2,
    judge_pair_both_orders,
)
from squid.eval.qwen_judge import query_judge as qwen_query_judge


ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "i5-pairwise-prompt-v1"
PROMPT_BUILDERS = {
    PROMPT_VERSION: build_pairwise_prompt,
    PAIRWISE_PROMPT_V2: build_pairwise_prompt_v2,
}
QUERY_FUNCTIONS = {
    "deepseek": deepseek_query_judge,
    "qwen": qwen_query_judge,
}
DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "qwen": "qwen4-max",
}
DEFAULT_OUTPUTS = {
    ("deepseek", "deepseek-chat"): (
        "data/reports/i5-pairwise-deepseek-chat-pilot.json"
    ),
    ("qwen", "qwen4-max"): (
        "data/reports/i5-pairwise-qwen4-max-pilot.json"
    ),
}




def write_report(path, report):
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=tuple(QUERY_FUNCTIONS),
        default="deepseek",
    )
    parser.add_argument(
        "--blind",
        default="data/eval/i5-human-calibration-blind.json",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prompt-version",
        choices=tuple(PROMPT_BUILDERS),
        default=PROMPT_VERSION,
    )
    parser.add_argument(
        "--qwen-max-tokens",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    args.model = args.model or DEFAULT_MODELS[args.provider]
    model_slug = args.model.replace("/", "-")
    args.output = args.output or DEFAULT_OUTPUTS.get(
        (args.provider, args.model),
        f"data/reports/i5-pairwise-{args.provider}-{model_slug}-pilot.json",
    )
    query_fn = QUERY_FUNCTIONS[args.provider]

    blind_path = ROOT / args.blind
    output_path = ROOT / args.output

    blind_raw = blind_path.read_bytes()
    blind = json.loads(blind_raw)

    assert blind["status"] == "labeled"
    assert all(
        pair["human"]["winner"] in {"A", "B", "tie"}
        for pair in blind["pairs"]
    )

    all_pairs = blind["pairs"]
    selected_pairs = (
        all_pairs[:args.limit]
        if args.limit is not None
        else all_pairs
    )

    blind_sha256 = hashlib.sha256(blind_raw).hexdigest()
    identity = {
        "schema_version": "i5-pairwise-judge-v1",
        "prompt_version": args.prompt_version,
        "provider": args.provider,
        "judge_model": args.model,
        "blind_sha256": blind_sha256,
        "query_settings": {
            "temperature": 0,
            "max_tokens": (
                args.qwen_max_tokens
                if args.provider == "qwen"
                else 16
            ),
        },
    }

    if output_path.exists():
        report = json.loads(
            output_path.read_text(encoding="utf-8")
        )
        if "provider" not in report and args.provider == "deepseek":
            report["provider"] = "deepseek"
        for key, value in identity.items():
            assert report.get(key) == value, (
                f"续跑身份不一致：{key}，"
                "请检查输入或使用新的 --output"
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
    selected_ids = {
        pair["pair_id"]
        for pair in selected_pairs
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

        if args.provider == "qwen":
            return query_fn(
                prompt,
                model=model,
                max_tokens=args.qwen_max_tokens,
            )

        return query_fn(prompt, model=model)

    for pair in selected_pairs:
        pair_id = pair["pair_id"]
        existing = saved_by_id.get(pair_id)

        def after_order(orders, pair_id=pair_id):
            saved_by_id[pair_id] = {
                "pair_id": pair_id,
                "judge_model": args.model,
                "orders": orders,
                "comparison": None,
            }
            persist("running")

        result = judge_pair_both_orders(
            pair,
            counting_query,
            model=args.model,
            existing=existing,
            after_order=after_order,
            prompt_builder=PROMPT_BUILDERS[args.prompt_version],
        )

        result["human_winner"] = pair["human"]["winner"]
        consensus = result["comparison"]["consensus_winner"]
        result["human_agreement"] = (
            consensus == result["human_winner"]
            if consensus is not None
            else None
        )

        saved_by_id[pair_id] = result
        persist("running")

    completed = [
        saved_by_id[pair_id]
        for pair_id in selected_ids
        if pair_id in saved_by_id
        and saved_by_id[pair_id].get("comparison") is not None
    ]
    comparable = [
        pair
        for pair in completed
        if pair["comparison"]["consensus_winner"] is not None
    ]

    report["summary"] = {
        "selected_pairs": len(selected_pairs),
        "completed_pairs": len(completed),
        "consistent": sum(
            pair["comparison"]["consistent"]
            for pair in completed
        ),
        "position_sensitive": sum(
            pair["comparison"]["position_sensitive"]
            for pair in completed
        ),
        "decisive_flip": sum(
            pair["comparison"]["decisive_flip"]
            for pair in completed
        ),
        "human_comparable": len(comparable),
        "human_agreement": sum(
            pair["human_agreement"]
            for pair in comparable
        ),
        "human_disagreement": sum(
            not pair["human_agreement"]
            for pair in comparable
        ),
    }
    report["boundary"] = (
        "这是少量分层试点样本的原始计数，不用于模型排名、"
        "统计相关性或普遍位置偏见结论。"
    )

    final_status = (
        "complete"
        if len(selected_pairs) == len(all_pairs)
        else "partial"
    )
    persist(final_status)

    print(json.dumps(
        report["summary"],
        ensure_ascii=False,
        indent=2,
    ))
    print(f"api_calls_this_run: {api_calls}")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
