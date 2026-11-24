import argparse
import csv
import json
import runpy
import subprocess
from collections import Counter
from pathlib import Path


def query_all(project):
    result = subprocess.run(
        [
            "defects4j",
            "query",
            "-p",
            project,
            "-q",
            (
                "bug.id,revision.id.buggy,revision.id.fixed,"
                "classes.modified,tests.trigger"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    rows = {}
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        rows[row[0]] = {
            "buggy_commit": row[1],
            "fixed_commit": row[2],
            "target_classes": row[3].split(";"),
            "trigger_tests": row[4].split(";"),
        }
    return rows


def classify_bug(file_audits):
    classes = {
        audit["classification"]
        for audit in file_audits
    }

    if not file_audits:
        return "no_source_patch"
    if len(file_audits) > 1:
        return "multi_file"
    if "invalid_or_unparsed_diff" in classes:
        return "invalid_or_unparsed_diff"
    if "no_source_difference" in classes:
        return "no_source_difference"
    if "multi_method" in classes or "contains_class_level_change" in classes:
        return "multi_location"
    if "class_level_only" in classes:
        return "class_level_only"
    if classes == {"single_method_candidate"}:
        return "single_method_candidate"
    return "needs_review"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--bug-ids", required=True)
    parser.add_argument("--workroot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkout-missing",
        action="store_true",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    audit_namespace = runpy.run_path(str(
        root / "scripts/i6_patch_audit.py"
    ))
    builder_namespace = runpy.run_path(str(
        root / "scripts/i6_build_case.py"
    ))
    find_java_file = builder_namespace["find_java_file"]
    java_source_roots = builder_namespace["JAVA_SOURCE_ROOTS"]
    audit_file = audit_namespace["audit_file"]

    metadata = query_all(args.project)
    bug_ids = args.bug_ids.split(",")
    workroot = Path(args.workroot)

    entries = []

    for bug_id in bug_ids:
        if bug_id not in metadata:
            entries.append({
                "instance_id": f"{args.project}-{bug_id}",
                "buggy_commit": None,
                "fixed_commit": None,
                "target_classes": [],
                "trigger_tests": [],
                "classification": "inactive_or_unknown",
                "files": [],
                "workdir": str(workroot / f"{args.project}-{bug_id}".lower()),
                "tests_verified_by_this_script": False,
            })
            continue

        bug = metadata[bug_id]
        instance_id = f"{args.project}-{bug_id}"
        workdir = workroot / instance_id.lower()

        workdir_missing = not (
            (workdir / "buggy").is_dir()
            and (workdir / "fixed").is_dir()
        )

        if workdir_missing and args.checkout_missing:
            workdir.mkdir(parents=True, exist_ok=True)

            for version, directory in (
                ("b", "buggy"),
                ("f", "fixed"),
            ):
                target = workdir / directory
                if target.is_dir():
                    continue

                subprocess.run(
                    [
                        "defects4j",
                        "checkout",
                        "-p",
                        args.project,
                        "-v",
                        f"{bug_id}{version}",
                        "-w",
                        str(target),
                    ],
                    check=True,
                )

            workdir_missing = not (
                (workdir / "buggy").is_dir()
                and (workdir / "fixed").is_dir()
            )

        if workdir_missing:
            entries.append({
                "instance_id": instance_id,
                "buggy_commit": bug["buggy_commit"],
                "fixed_commit": bug["fixed_commit"],
                "target_classes": bug["target_classes"],
                "trigger_tests": bug["trigger_tests"],
                "classification": "missing_workdir",
                "files": [],
                "workdir": str(workdir),
                "tests_verified_by_this_script": False,
            })
            continue


        file_audits = []
        errors = []

        try:
            for target_class in bug["target_classes"]:
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
                file_audit = audit_file(
                    buggy_file,
                    fixed_file,
                )
                file_audit["source_file"] = (
                    buggy_file.relative_to(
                        workdir / "buggy"
                    ).as_posix()
                )
                file_audits.append(file_audit)

            classification = classify_bug(file_audits)
        except (OSError, RuntimeError, ValueError) as error:
            classification = "audit_error"
            file_audits = []
            errors.append(str(error))

        entries.append({
            "instance_id": instance_id,
            "buggy_commit": bug["buggy_commit"],
            "fixed_commit": bug["fixed_commit"],
            "target_classes": bug["target_classes"],
            "trigger_tests": bug["trigger_tests"],
            "classification": classification,
            "files": file_audits,
            "workdir": str(workdir),
            "errors": errors,
            "tests_verified_by_this_script": False,
        })


    report = {
        "schema_version": "i6-batch-audit-v1",
        "project": args.project,
        "summary": {
            "total": len(entries),
            "classifications": dict(Counter(
                entry["classification"]
                for entry in entries
            )),
            "single_method_candidates": [
                entry["instance_id"]
                for entry in entries
                if entry["classification"] == "single_method_candidate"
            ],
        },
        "entries": entries,
        "boundary": (
            "这是 Patch 结构审计，不代表编译测试已验收，"
            "也不自动决定训练资格。"
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
