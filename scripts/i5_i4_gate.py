import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "original"
CANDIDATES = ("cot_sft_v2", "dpo_v1", "grpo_one_group")
MODELS = (BASELINE, *CANDIDATES)


def validate_sources(ci_report, root=ROOT):
    aligned_indices = None
    cases_per_model = None

    for model, source in ci_report["sources"].items():
        path = root / source["path"]
        raw = path.read_bytes()

        assert hashlib.sha256(raw).hexdigest() == source["sha256"], (
            f"{model} 源报告哈希不匹配"
        )

        data = json.loads(raw)
        cases = data["cases"]
        indices = [case["index"] for case in cases]

        assert len(indices) == len(set(indices)), (
            f"{model} 存在重复 case index"
        )

        if aligned_indices is None:
            aligned_indices = set(indices)
            cases_per_model = len(cases)
        else:
            assert set(indices) == aligned_indices
            assert len(cases) == cases_per_model

        assert ci_report["models"][model]["total"] == len(cases)

    assert set(ci_report["sources"]) == set(MODELS)
    assert set(ci_report["models"]) == set(MODELS)

    expected_pairs = {
        f"{left} vs {right}"
        for left, right in itertools.combinations(MODELS, 2)
    }
    assert set(ci_report["pairwise"]) == expected_pairs

    for result in ci_report["pairwise"].values():
        bootstrap = result["bootstrap"]
        assert bootstrap["observed"]["total"] == cases_per_model
        assert bootstrap["intervals"]["net_win_rate"] is not None

    return {
        "models": len(MODELS),
        "cases_per_model": cases_per_model,
        "aligned_indices": len(aligned_indices),
        "pairwise_comparisons": len(expected_pairs),
    }


def candidate_result(pair_name, result):
    effect = result["effect_from_net_interval"]
    status = {
        "left_better": "regression_detected",
        "left_worse": "improvement_detected",
        "inconclusive": "inconclusive",
    }[effect]

    bootstrap = result["bootstrap"]
    return {
        "evidence_pair": pair_name,
        "effect_for_original_vs_candidate": effect,
        "status": status,
        "net_win_rate": bootstrap["observed"]["net_win_rate"],
        "interval": bootstrap["intervals"]["net_win_rate"],
    }


def build_gate(ci_report, coverage):
    candidates = {}

    for candidate in CANDIDATES:
        pair_name = f"{BASELINE} vs {candidate}"
        candidates[candidate] = candidate_result(
            pair_name,
            ci_report["pairwise"][pair_name],
        )

    statuses = {
        candidate["status"]
        for candidate in candidates.values()
    }
    regression_detected = "regression_detected" in statuses

    if regression_detected:
        evidence_status = "regression_detected"
    elif statuses == {"improvement_detected"}:
        evidence_status = "improvement_detected"
    else:
        evidence_status = "inconclusive"

    return {
        "schema_version": "i5-i4-gate-v1",
        "artifact_status": "pass",
        "coverage": coverage,
        "baseline": BASELINE,
        "candidates": candidates,
        "regression_detected": regression_detected,
        "evidence_status": evidence_status,
        "boundary": (
            "artifact_status=pass 只表示报告与源数据有效。"
            "inconclusive 不证明无回退、无提升或模型等效。"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--ci-report",
        default="data/reports/i4-final-comparison-ci.json",
    )
    parser.add_argument(
        "--output",
        default="data/reports/i5-i4-gate.json",
    )
    args = parser.parse_args()

    ci_path = ROOT / args.ci_report
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/i5_i4_bootstrap_report.py"),
            "--resamples",
            str(args.resamples),
            "--seed",
            str(args.seed),
            "--confidence",
            str(args.confidence),
            "--output",
            str(ci_path),
        ],
        cwd=ROOT,
        check=True,
    )

    ci_report = json.loads(ci_path.read_text(encoding="utf-8"))
    coverage = validate_sources(ci_report)
    gate = build_gate(ci_report, coverage)

    output_path = ROOT / args.output
    output_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"artifact_status: {gate['artifact_status']}")
    print(f"coverage: {gate['coverage']}")

    for name, result in gate["candidates"].items():
        interval = result["interval"]
        print(
            f"{name}: {result['status']}, "
            f"net={result['net_win_rate']:.6f}, "
            f"CI=[{interval['low']:.6f}, "
            f"{interval['high']:.6f}]"
        )

    print(f"regression_detected: {gate['regression_detected']}")
    print(f"evidence_status: {gate['evidence_status']}")
    print(f"report: {output_path}")

    return 2 if gate["regression_detected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
