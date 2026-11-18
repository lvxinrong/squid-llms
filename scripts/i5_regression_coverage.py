import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ENTRIES = [
    {
        "stage": "I0",
        "name": "numerical_alignment",
        "coverage_class": "controlled_alignment",
        "paths": [
            "data/reports/i0-numerical-alignment.json"
        ],
        "required_fields": [
            "config",
            "summary",
            "full_logits",
            "cache",
        ],
    },
    {
        "stage": "I1",
        "name": "heldout_bpb_smoke",
        "coverage_class": "single_observation",
        "paths": ["data/reports/i1-bpb-smoke.json"],
        "required_fields": ["evaluated_tokens", "bpb"],
    },
    {
        "stage": "I2",
        "name": "fp32_three_runs",
        "coverage_class": "repeated_measurement",
        "paths": [
            "data/reports/i2-fp32-baseline-run-1.json",
            "data/reports/i2-fp32-baseline-run-2.json",
            "data/reports/i2-fp32-baseline-run-3.json",
        ],
        "required_fields": [
            "mode",
            "device",
            "dtype",
            "measured_steps",
            "tokens_per_second",
            "losses",
        ],
    },
    {
        "stage": "I2",
        "name": "compile_three_runs",
        "coverage_class": "repeated_measurement",
        "paths": [
            "data/reports/i2-compile-fp32-baseline-run-1.json",
            "data/reports/i2-compile-fp32-baseline-run-2.json",
            "data/reports/i2-compile-fp32-baseline-run-3.json",
        ],
        "required_fields": [
            "mode",
            "device",
            "dtype",
            "measured_steps",
            "tokens_per_second",
            "losses",
        ],
    },
    {
        "stage": "I3",
        "name": "cache_three_runs",
        "coverage_class": "repeated_measurement",
        "paths": [
            "data/reports/i3-inference-baseline-cache-mps-run-1.json",
            "data/reports/i3-inference-baseline-cache-mps-run-2.json",
            "data/reports/i3-inference-baseline-cache-mps-run-3.json",
        ],
        "required_fields": [
            "use_cache",
            "ttft_seconds",
            "decode_tokens_per_second",
            "generated_text",
        ],
    },
    {
        "stage": "I3",
        "name": "no_cache_three_runs",
        "coverage_class": "repeated_measurement",
        "paths": [
            "data/reports/i3-inference-baseline-no-cache-mps-run-1.json",
            "data/reports/i3-inference-baseline-no-cache-mps-run-2.json",
            "data/reports/i3-inference-baseline-no-cache-mps-run-3.json",
        ],
        "required_fields": [
            "use_cache",
            "ttft_seconds",
            "decode_tokens_per_second",
            "generated_text",
        ],
    },
    {
        "stage": "I3",
        "name": "speculative_fp32_k2",
        "coverage_class": "single_workload",
        "paths": [
            "data/reports/i3-speculative-qwen-fp32-k2.json"
        ],
        "required_fields": [
            "output_token_ids_match",
            "speedup_reference_over_speculative",
            "draft_acceptance_rate",
            "reference",
            "speculative",
        ],
    },
    {
        "stage": "I4",
        "name": "paired_bootstrap_gate",
        "coverage_class": "paired_gate",
        "paths": ["data/reports/i5-i4-gate.json"],
        "required_fields": [
            "artifact_status",
            "coverage",
            "candidates",
            "evidence_status",
        ],
    },
]


def check_entry(entry, root):
    result = {
        "stage": entry["stage"],
        "name": entry["name"],
        "coverage_class": entry["coverage_class"],
        "artifact_status": "pass",
        "artifacts": [],
        "errors": [],
    }

    if entry["coverage_class"] == "needs_export":
        result["artifact_status"] = "not_applicable"
        return result

    for relative_path in entry["paths"]:
        path = root / relative_path

        try:
            raw = path.read_bytes()
            document = json.loads(raw)
            missing = [
                field
                for field in entry["required_fields"]
                if field not in document
            ]
            if missing:
                raise ValueError(
                    f"缺少字段：{missing}"
                )

            result["artifacts"].append({
                "path": relative_path,
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        except (OSError, ValueError, json.JSONDecodeError) as error:
            result["artifact_status"] = "fail"
            result["errors"].append(
                f"{relative_path}: {error}"
            )

    return result


def build_coverage_report(entries, root=ROOT):
    checked = [
        check_entry(entry, root)
        for entry in entries
    ]
    classes = Counter(
        entry["coverage_class"]
        for entry in checked
    )
    invalid = [
        entry
        for entry in checked
        if entry["artifact_status"] == "fail"
    ]

    return {
        "schema_version": "i5-regression-coverage-v1",
        "mode": "cached",
        "artifact_status": "fail" if invalid else "pass",
        "coverage_status": (
            "incomplete"
            if classes["needs_export"]
            else "complete_for_registered_scope"
        ),
        "registered_scope_only": True,
        "summary": {
            "entries": len(checked),
            "coverage_classes": dict(classes),
            "invalid_entries": len(invalid),
        },
        "entries": checked,
        "boundary": (
            "只验证已注册历史产物的存在与字段契约，"
            "不证明整个里程碑能力已完整自动回归，也不重新运行基准。"
        ),
    }


def main():
    report = build_coverage_report(ENTRIES)
    output_path = ROOT / "data/reports/i5-regression-coverage.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for entry in report["entries"]:
        print(
            f"{entry['stage']} {entry['name']}: "
            f"class={entry['coverage_class']}, "
            f"artifacts={entry['artifact_status']}, "
            f"files={len(entry['artifacts'])}"
        )

    print(json.dumps(
        report["summary"],
        ensure_ascii=False,
        indent=2,
    ))
    print(f"artifact_status: {report['artifact_status']}")
    print(f"coverage_status: {report['coverage_status']}")
    print(f"report: {output_path}")

    return 1 if report["artifact_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
