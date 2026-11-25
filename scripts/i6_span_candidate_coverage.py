import argparse
import json
import runpy
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOW_LIMITS = (1, 2, 4, 8, 16)


def build_spans(source, max_lines):
    lines = source.splitlines()
    spans = []

    for start in range(len(lines)):
        max_end = min(len(lines), start + max_lines)

        for end in range(start + 1, max_end + 1):
            spans.append({
                "span_id": f"S{len(spans) + 1:04d}",
                "start_line": start + 1,
                "end_line": end,
                "text": "\n".join(lines[start:end]),
            })

    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))

    candidate_sets = {
        limit: build_spans(case["buggy_source"], limit)
        for limit in WINDOW_LIMITS
    }

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    namespace = runpy.run_path(str(
        ROOT / "scripts/i6_structured_sft_data.py"
    ))
    target_method = namespace["re_target_method"](
        case["buggy_source"]
    )
    reference_edit = namespace["reference_to_edit"](
        key["reference_patch"],
        target_method,
        case["buggy_source"],
    )
    old_text = reference_edit["old_text"]

    results = []

    for limit, spans in candidate_sets.items():
        matches = [
            span
            for span in spans
            if span["text"] == old_text
        ]
        text_counts = Counter(
            span["text"]
            for span in spans
        )

        matched_span = None
        if len(matches) == 1:
            matched_span = {
                key: matches[0][key]
                for key in ("span_id", "start_line", "end_line")
            }

        results.append({
            "max_lines": limit,
            "candidate_count": len(spans),
            "covered": len(matches) == 1,
            "match_count": len(matches),
            "matched_span": matched_span,
            "duplicate_text_groups": sum(
                1
                for count in text_counts.values()
                if count > 1
            ),
        })

    report = {
        "instance_id": case["instance_id"],
        "candidate_construction": "contiguous_buggy_source_windows",
        "candidate_source": "buggy_source_only",
        "source_line_count": len(case["buggy_source"].splitlines()),
        "reference_span_lines": len(old_text.splitlines()),
        "reference_used_for_coverage_only": True,
        "reference_new_text_used": False,
        "results": results,
    }

    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(case["instance_id"])
    for result in results:
        print(
            "  max_lines=", result["max_lines"],
            "candidates=", result["candidate_count"],
            "covered=", result["covered"],
            "matched=", result["matched_span"],
        )
    print(f"report: {output}")


if __name__ == "__main__":
    main()
