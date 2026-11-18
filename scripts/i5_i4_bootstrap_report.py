import argparse
import hashlib
import itertools
import json
import runpy
from pathlib import Path

from squid.eval.bootstrap import paired_bootstrap


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = runpy.run_path(str(
    ROOT / "scripts/i4_final_comparison.py"
))
REPORTS = NAMESPACE["REPORTS"]
load_report = NAMESPACE["load_report"]
compare = NAMESPACE["compare"]


def classify_net_effect(interval):
    if interval is None:
        return "unknown"
    if interval["low"] > 0:
        return "left_better"
    if interval["high"] < 0:
        return "left_worse"
    return "inconclusive"


def compare_with_bootstrap(
    left_cases,
    right_cases,
    n_resamples,
    seed,
    confidence,
):
    result = compare(left_cases, right_cases)
    bootstrap = paired_bootstrap(
        left_cases,
        right_cases,
        n_resamples=n_resamples,
        seed=seed,
        confidence=confidence,
    )
    result["effect_from_net_interval"] = classify_net_effect(
        bootstrap["intervals"]["net_win_rate"]
    )
    result["bootstrap"] = bootstrap
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--output",
        default="data/reports/i4-final-comparison-ci.json",
    )
    args = parser.parse_args()

    loaded = {
        name: load_report(filename)
        for name, filename in REPORTS.items()
    }
    sources = {}

    for name, filename in REPORTS.items():
        raw = (ROOT / "data/reports" / filename).read_bytes()
        sources[name] = {
            "path": f"data/reports/{filename}",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    pairwise = {}

    for left_name, right_name in itertools.combinations(REPORTS, 2):
        comparison_name = f"{left_name} vs {right_name}"
        pairwise[comparison_name] = compare_with_bootstrap(
            loaded[left_name][0]["cases"],
            loaded[right_name][0]["cases"],
            n_resamples=args.resamples,
            seed=args.seed,
            confidence=args.confidence,
        )

    report = {
        "schema_version": "i5-i4-bootstrap-v1",
        "sources": sources,
        "models": {
            name: summary
            for name, (_, summary) in loaded.items()
        },
        "pairwise": pairwise,
        "definitions": {
            "net_win_rate": "(wins - losses) / 全部 case 数",
            "effect_from_net_interval": (
                "net_win_rate 区间严格大于 0 为 left_better，"
                "严格小于 0 为 left_worse，其余为 inconclusive"
            ),
        },
        "boundary": (
            "这是同一 32 条合成 CoT 题上的规则评委配对 bootstrap，"
            "不是开放域能力、LLM 评委胜率或普遍模型优劣证明。"
        ),
    }

    output_path = ROOT / args.output
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for name, result in pairwise.items():
        observed = result["bootstrap"]["observed"]
        interval = result["bootstrap"]["intervals"]["net_win_rate"]
        print(
            f"{name}: "
            f"W/L/T={observed['wins']}/"
            f"{observed['losses']}/{observed['ties']}, "
            f"net={observed['net_win_rate']:.6f}, "
            f"CI=[{interval['low']:.6f}, "
            f"{interval['high']:.6f}], "
            f"effect={result['effect_from_net_interval']}"
        )

    print(f"resamples: {args.resamples}")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
