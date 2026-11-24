import argparse
import json
import runpy
from collections import Counter
from pathlib import Path


def choose_method_marker(file_audit, lines, extract_java_block, method_declarations):
    changed = [
        item
        for item in file_audit["changed_methods"]
        if item != "class_level_or_unknown"
    ]
    if len(changed) != 1:
        raise ValueError("不是唯一的代码方法变更")

    kind, name = changed[0].split(":", 1)
    matches = []

    for declaration in method_declarations(lines):
        if (
            declaration["kind"] != kind
            or declaration["name"] != name
        ):
            continue

        marker = lines[declaration["index"]].strip()
        block = extract_java_block(lines, marker)

        for hunk in file_audit["hunks"]:
            hunk_start = hunk["old_start"]
            hunk_end = hunk_start + max(hunk["old_count"] - 1, 0)

            if (
                block["start_line"] <= hunk_end
                and block["end_line"] >= hunk_start
            ):
                matches.append(marker)
                break

    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError(
            f"无法唯一确定方法签名：{kind}:{name}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--exclude", default="")
    parser.add_argument(
        "--data-role",
        choices=("train_candidate", "validation_candidate", "eval_holdout",),
        default="train_candidate",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    builder_namespace = runpy.run_path(str(
        root / "scripts/i6_build_case.py"
    ))
    audit_namespace = runpy.run_path(str(
        root / "scripts/i6_patch_audit.py"
    ))

    extract_java_block = builder_namespace["extract_java_block"]
    reference_diff = builder_namespace["reference_diff"]
    method_declarations = audit_namespace["method_declarations"]

    find_java_file = builder_namespace["find_java_file"]
    java_source_roots = builder_namespace["JAVA_SOURCE_ROOTS"]
    java_test_roots = builder_namespace["JAVA_TEST_ROOTS"]

    report = json.loads(
        Path(args.audit_report).read_text(encoding="utf-8")
    )
    excluded = set(args.exclude.split(",")) - {""}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for entry in report["entries"]:
        instance_id = entry["instance_id"]
        result = {
            "instance_id": instance_id,
            "status": "created",
            "errors": [],
        }

        if entry["classification"] != "single_method_candidate":
            result["status"] = "skipped_not_single_method"
            results.append(result)
            continue

        if instance_id in excluded:
            result["status"] = "excluded_heldout"
            results.append(result)
            continue

        case_path = output_dir / f"{instance_id.lower()}-case.json"
        key_path = output_dir / f"{instance_id.lower()}-key.json"

        if case_path.exists() or key_path.exists():
            result["status"] = "already_exists"
            results.append(result)
            continue

        if len(entry["target_classes"]) != 1:
            result["status"] = "skipped_multi_class"
            results.append(result)
            continue

        try:
            target_class = entry["target_classes"][0]
            workdir = Path(entry["workdir"])
            buggy_file = find_java_file(
                workdir / "buggy",
                target_class,
                java_source_roots,
            )
            fixed_file = find_java_file(
                workdir / "fixed",
                target_class,
                java_source_roots,
            )
            source_file = buggy_file.relative_to(
                workdir / "buggy"
            ).as_posix()


            buggy_lines = buggy_file.read_text(
                encoding="utf-8"
            ).splitlines()

            if len(entry["files"]) != 1:
                raise ValueError("当前只支持单文件单方法候选")

            file_audit = entry["files"][0]
            marker = choose_method_marker(
                file_audit,
                buggy_lines,
                extract_java_block,
                method_declarations,
            )
            source = extract_java_block(buggy_lines, marker)

            trigger_test = entry["trigger_tests"][0]
            test_class = trigger_test.split("::", 1)[0]
            test_method = trigger_test.rsplit("::", 1)[-1]
            test_file = find_java_file(
                workdir / "buggy",
                test_class,
                java_test_roots,
            )
            test_lines = test_file.read_text(
                encoding="utf-8"
            ).splitlines()
            test = extract_java_block(
                test_lines,
                f"public void {test_method}",
            )

            case = {
                "instance_id": instance_id,
                "project": report["project"],
                "buggy_commit": entry["buggy_commit"],
                "fixed_commit": entry["fixed_commit"],
                "target_class": target_class,
                "trigger_test": trigger_test,
                "source_file": source_file,
                "source_span": {
                    "start_line": source["start_line"],
                    "end_line": source["end_line"],
                },
                "buggy_source": source["text"],
                "test_source": test["text"],
                "data_role": args.data_role,
                "excluded_from_training": (
                    args.data_role != "train_candidate"
                ),
                "evaluation": {
                    "compile": True,
                    "trigger_test": "fail_to_pass",
                    "regression_tests": "pass",
                    "decoding": "greedy",
                    "pass_at": 1,
                },
                "reference_patch_in_prompt": False,
            }
            key = {
                "instance_id": instance_id,
                "reference_patch": reference_diff(
                    buggy_file,
                    fixed_file,
                ),
            }

            case_path.write_text(
                json.dumps(case, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            key_path.write_text(
                json.dumps(key, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result["case"] = str(case_path)
            result["method_marker"] = marker

        except (OSError, ValueError, StopIteration) as error:
            result["status"] = "needs_review"
            result["errors"].append(str(error))

        results.append(result)

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    summary = dict(Counter(
        result["status"]
        for result in results
    ))
    output_report.write_text(
        json.dumps({
            "schema_version": "i6-batch-build-cases-v1",
            "source_audit": args.audit_report,
            "summary": summary,
            "results": results,
            "boundary": (
                "只批量生成样本，不训练模型；needs_review 不自动进入训练集。"
            ),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {output_report}")


if __name__ == "__main__":
    main()
