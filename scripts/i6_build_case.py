import argparse
import csv
import json
import subprocess
from pathlib import Path


def query_metadata(
    project,
    bug_id,
    test_method=None,
    trigger_test_override=None,
):
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

    for row in csv.reader(result.stdout.splitlines()):
        if not row or row[0] != str(bug_id):
            continue

        trigger_test = row[4]

        if trigger_test_override:
            candidates = trigger_test.split(";")
            if trigger_test_override not in candidates:
                raise ValueError(
                    f"指定触发测试不在元数据中：{trigger_test_override}"
                )
            trigger_test = trigger_test_override
        elif test_method:
            candidates = trigger_test.split(";")
            matches = [
                candidate
                for candidate in candidates
                if candidate.rsplit("::", 1)[-1] == test_method
            ]
            if not matches:
                raise ValueError(
                    f"没有找到名为 {test_method} 的触发测试"
                )
            if len(matches) > 1:
                raise ValueError(
                    "存在多个同名测试，请改用完整的 --trigger-test"
                )
            trigger_test = matches[0]


        if ";" in row[3]:
            raise ValueError("第一版只支持单个目标修改类")

        return {
            "instance_id": f"{project}-{bug_id}",
            "project": project,
            "buggy_commit": row[1],
            "fixed_commit": row[2],
            "target_class": row[3],
            "trigger_test": trigger_test,
        }

    raise ValueError(f"没有找到 {project}-{bug_id}")


def extract_java_block(lines, marker):
    start = next(
        index
        for index, line in enumerate(lines)
        if marker in line
    )

    depth = 0
    for index in range(start, len(lines)):
        depth += lines[index].count("{")
        depth -= lines[index].count("}")
        if depth == 0:
            return {
                "text": "\n".join(lines[start:index + 1]),
                "start_line": start + 1,
                "end_line": index + 1,
            }

    raise ValueError(f"方法没有找到完整结束括号：{marker}")


def reference_diff(buggy_file, fixed_file):
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-index",
            "--",
            str(buggy_file),
            str(fixed_file),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--bug-id", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--method-marker", required=True)
    parser.add_argument("--test-method", default=None)
    parser.add_argument(
        "--data-role",
        choices=("train_candidate", "validation_candidate", "eval_holdout"),
        default="eval_holdout",
    )
    parser.add_argument("--trigger-test", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    metadata = query_metadata(
        args.project,
        args.bug_id,
        test_method=args.test_method,
        trigger_test_override=args.trigger_test,
    )

    source_file = (
        f"src/main/java/"
        f"{metadata['target_class'].replace('.', '/')}.java"
    )
    test_class = metadata["trigger_test"].split("::", 1)[0]
    test_file = f"src/test/java/{test_class.replace('.', '/')}.java"

    buggy_file = workdir / "buggy" / source_file
    fixed_file = workdir / "fixed" / source_file
    trigger_file = workdir / "buggy" / test_file

    buggy_lines = buggy_file.read_text(
        encoding="utf-8"
    ).splitlines()
    test_lines = trigger_file.read_text(
        encoding="utf-8"
    ).splitlines()

    source = extract_java_block(
        buggy_lines,
        args.method_marker,
    )
    test_name = metadata["trigger_test"].rsplit("::", 1)[-1]
    test = extract_java_block(
        test_lines,
        f"public void {test_name}",
    )

    case = {
        **metadata,
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
        "instance_id": metadata["instance_id"],
        "reference_patch": reference_diff(
            buggy_file,
            fixed_file,
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = metadata["instance_id"].lower()

    case_path = output_dir / f"{prefix}-case.json"
    key_path = output_dir / f"{prefix}-key.json"

    if case_path.exists() or key_path.exists():
        raise FileExistsError(
            f"样本已存在，拒绝覆盖：{case_path} / {key_path}"
        )

    case_path.write_text(
        json.dumps(case, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    key_path.write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"case: {case_path}")
    print(f"key: {key_path}")
    print(f"data_role: {args.data_role}")
    print(f"source_span: {case['source_span']}")


if __name__ == "__main__":
    main()
