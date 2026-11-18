import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(root, relative_path):
    return json.loads(
        (root / relative_path).read_text(encoding="utf-8")
    )


def summarize_training_runs(root, paths):
    documents = [
        load_json(root, path)
        for path in paths
    ]
    throughput = [
        document["tokens_per_second"]
        for document in documents
    ]

    return {
        "runs": len(documents),
        "tokens_per_second": {
            "min": min(throughput),
            "median": statistics.median(throughput),
            "max": max(throughput),
        },
        "all_losses_finite": all(
            math.isfinite(loss)
            for document in documents
            for loss in document["losses"]
        ),
        "median_tokens_per_second": statistics.median(throughput),
    }


def determine_overall(
    artifact_status,
    correctness_checks,
    regression_detected,
    capability_evidence,
):
    if artifact_status != "pass" or not all(correctness_checks.values()):
        return "failed", 1
    if regression_detected:
        return "regression_detected", 2
    if capability_evidence == "inconclusive":
        return "artifacts_valid_capability_inconclusive", 0
    return "artifacts_valid", 0


def build_summary(root, refresh_controlled):
    coverage = load_json(
        root,
        "data/reports/i5-regression-coverage.json",
    )
    i0 = load_json(
        root,
        "data/reports/i0-numerical-alignment.json",
    )
    i1 = load_json(
        root,
        "data/reports/i1-bpb-smoke.json",
    )
    i4 = load_json(
        root,
        "data/reports/i5-i4-gate.json",
    )

    fp32 = summarize_training_runs(
        root,
        [
            "data/reports/i2-fp32-baseline-run-1.json",
            "data/reports/i2-fp32-baseline-run-2.json",
            "data/reports/i2-fp32-baseline-run-3.json",
        ],
    )
    compile_run = summarize_training_runs(
        root,
        [
            "data/reports/i2-compile-fp32-baseline-run-1.json",
            "data/reports/i2-compile-fp32-baseline-run-2.json",
            "data/reports/i2-compile-fp32-baseline-run-3.json",
        ],
    )
    compile_relative = (
        compile_run["median_tokens_per_second"]
        / fp32["median_tokens_per_second"]
        - 1
    )

    cache_documents = [
        load_json(
            root,
            f"data/reports/i3-inference-baseline-cache-mps-run-{index}.json",
        )
        for index in (1, 2, 3)
    ]
    no_cache_documents = [
        load_json(
            root,
            f"data/reports/i3-inference-baseline-no-cache-mps-run-{index}.json",
        )
        for index in (1, 2, 3)
    ]
    cache_decode = statistics.median(
        document["decode_tokens_per_second"]
        for document in cache_documents
    )
    no_cache_decode = statistics.median(
        document["decode_tokens_per_second"]
        for document in no_cache_documents
    )
    greedy_output_match = len({
        document["generated_text"]
        for document in [*cache_documents, *no_cache_documents]
    }) == 1

    speculative = load_json(
        root,
        "data/reports/i3-speculative-qwen-fp32-k2.json",
    )

    for summary in (fp32, compile_run):
        summary.pop("median_tokens_per_second")

    stages = {
        "I0": {
            "status": i0["summary"]["status"],
            "full_logits": (
                f"{i0['summary']['full_logits_passed']}/"
                f"{i0['summary']['full_logits_cases']}"
            ),
            "cache_decode": (
                f"{i0['summary']['cache_decode_passed']}/"
                f"{i0['summary']['cache_decode_steps']}"
            ),
            "scope": i0["boundary"],
        },
        "I1": {
            "bpb": i1["bpb"],
            "evaluated_tokens": i1["evaluated_tokens"],
            "coverage_class": "single_observation",
        },
        "I2": {
            "fp32": fp32,
            "compile": compile_run,
            "compile_relative_throughput": compile_relative,
            "statistics": "三次运行的中位数与范围，不提供性能 CI",
        },
        "I3": {
            "cache_decode_tokens_per_second_median": cache_decode,
            "no_cache_decode_tokens_per_second_median": no_cache_decode,
            "cache_decode_speedup": cache_decode / no_cache_decode,
            "greedy_output_match": greedy_output_match,
            "speculative_k2": {
                "output_token_ids_match": speculative[
                    "output_token_ids_match"
                ],
                "speedup": speculative[
                    "speedup_reference_over_speculative"
                ],
                "draft_acceptance_rate": speculative[
                    "draft_acceptance_rate"
                ],
            },
        },
        "I4": {
            "artifact_status": i4["artifact_status"],
            "evidence_status": i4["evidence_status"],
            "regression_detected": i4["regression_detected"],
            "candidates": {
                name: result["status"]
                for name, result in i4["candidates"].items()
            },
        },
    }

    correctness_checks = {
        "I0_controlled_alignment": i0["summary"]["status"] == "pass",
        "I2_losses_finite": (
            fp32["all_losses_finite"]
            and compile_run["all_losses_finite"]
        ),
        "I3_cache_greedy_output_match": greedy_output_match,
        "I3_speculative_output_match": speculative[
            "output_token_ids_match"
        ],
        "I4_artifacts_valid": i4["artifact_status"] == "pass",
    }
    overall_status, exit_code = determine_overall(
        coverage["artifact_status"],
        correctness_checks,
        i4["regression_detected"],
        i4["evidence_status"],
    )

    return {
        "schema_version": "i5-regression-summary-v1",
        "overview": {
            "overall_status": overall_status,
            "exit_code": exit_code,
            "artifact_status": coverage["artifact_status"],
            "coverage_status": coverage["coverage_status"],
            "correctness_checks": correctness_checks,
            "capability_evidence": i4["evidence_status"],
            "regression_detected": i4["regression_detected"],
            "refresh_scope": (
                ["I0", "I4"]
                if refresh_controlled
                else []
            ),
            "full_live_refresh": False,
        },
        "stages": stages,
        "boundary": (
            "当前注册范围的统一报告。I1～I3 始终读取历史测量；"
            "refresh-controlled 只刷新 I0/I4。"
            "性能结果不做小样本 CI，I4 inconclusive 不代表等效。"
        ),
    }, exit_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-controlled", action="store_true")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()

    if args.refresh_controlled:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/i0_numerical_alignment_report.py"),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/i5_i4_gate.py"),
                "--resamples",
                str(args.resamples),
                "--seed",
                str(args.seed),
            ],
            cwd=ROOT,
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/i5_regression_coverage.py"),
        ],
        cwd=ROOT,
        check=True,
    )

    report, exit_code = build_summary(
        ROOT,
        args.refresh_controlled,
    )
    output_path = ROOT / "data/reports/i5-regression-summary.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        report["overview"],
        ensure_ascii=False,
        indent=2,
    ))
    print(json.dumps(
        report["stages"]["I2"],
        ensure_ascii=False,
        indent=2,
    ))
    print(json.dumps(
        report["stages"]["I3"],
        ensure_ascii=False,
        indent=2,
    ))
    print(f"report: {output_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
