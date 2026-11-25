import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


def find_method_braces(text):
    state = "code"
    open_index = None
    depth = 0
    index = 0

    while index < len(text):
        char = text[index]
        next_char = (
            text[index + 1]
            if index + 1 < len(text)
            else ""
        )

        if state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and next_char == "/":
                state = "code"
                index += 1
        elif state in {"string", "char"}:
            if char == "\\":
                index += 1
            elif (
                state == "string"
                and char == '"'
            ) or (
                state == "char"
                and char == "'"
            ):
                state = "code"
        else:
            if char == "/" and next_char == "/":
                state = "line_comment"
                index += 1
            elif char == "/" and next_char == "*":
                state = "block_comment"
                index += 1
            elif char == '"':
                state = "string"
            elif char == "'":
                state = "char"
            elif char == "{":
                if open_index is None:
                    open_index = index
                    depth = 1
                else:
                    depth += 1
            elif char == "}" and open_index is not None:
                depth -= 1
                if depth == 0:
                    return open_index, index

        index += 1

    raise ValueError("方法缺少完整、匹配的花括号")


def validate_method_response(response, buggy_method):
    first_content = re.search(r"\S", response)
    if first_content is None:
        raise ValueError("候选为空")

    open_index, close_index = find_method_braces(response)
    original_open = buggy_method.find("{")
    if original_open < 0:
        raise ValueError("原方法缺少声明花括号")

    expected_signature = "".join(
        buggy_method[:original_open].split()
    )
    actual_signature = "".join(
        response[first_content.start():open_index].split()
    )

    if actual_signature != expected_signature:
        raise ValueError(
            "候选不是签名匹配的目标方法："
            f"{actual_signature}"
        )

    if any(
        not char.isspace()
        for char in response[close_index + 1:]
    ):
        raise ValueError("目标方法之后存在额外声明或内容")

    return {
        "method_start": first_content.start(),
        "method_end": close_index + 1,
    }


def replace_exact_method(file_text, buggy_method, response):
    if file_text.count(buggy_method) != 1:
        raise ValueError("完整 buggy 方法在原文件中不唯一")

    start = file_text.index(buggy_method)
    end = start + len(buggy_method)
    patched = file_text[:start] + response + file_text[end:]

    assert patched[:start] == file_text[:start]
    assert patched[start + len(response):] == file_text[end:]
    assert patched[start:start + len(response)] == response
    return patched


def changes_outside_given_old_text(
    response,
    buggy_method,
    old_text,
):
    if buggy_method.count(old_text) != 1:
        raise ValueError("给定 old_text 在原方法中不唯一")

    start = buggy_method.index(old_text)
    prefix = buggy_method[:start]
    suffix = buggy_method[start + len(old_text):]

    core_end = len(response)
    while (
        core_end > 0
        and response[core_end - 1].isspace()
    ):
        core_end -= 1
    core = response[:core_end]

    return not (
        core.startswith(prefix)
        and core.endswith(suffix)
    )


def run_defects4j(workdir):
    compile_result = subprocess.run(
        ["defects4j", "compile"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    test_result = subprocess.run(
        ["defects4j", "test"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    test_output = test_result.stdout + test_result.stderr
    test_execution_observed = (
        "Running ant (run.dev.tests)" in test_output
    )
    tests_passed = (
        compile_result.returncode == 0
        and test_result.returncode == 0
        and test_execution_observed
        and "Failing tests: 0" in test_output
    )

    return {
        "compile_returncode": compile_result.returncode,
        "compile_output": (
            compile_result.stdout + compile_result.stderr
        ),
        "test_returncode": test_result.returncode,
        "test_output": test_output,
        "test_execution_observed": test_execution_observed,
        "developer_tests_passed": tests_passed,
        "failing_tests_detected": bool(re.search(
            r"Failing tests: [1-9]\d*",
            test_output,
        )),
    }


def write_result(path, result):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value
        for key, value in result.items()
        if key not in {
            "compile_output",
            "test_output",
            "buggy_compile_output",
            "buggy_test_output",
        }
    }, ensure_ascii=False, indent=2))
    print(f"report: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--verify-buggy-baseline",
        action="store_true",
    )
    args = parser.parse_args()

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    candidate_report = json.loads(
        Path(args.candidate_report).read_text(encoding="utf-8")
    )
    response = candidate_report["response"]
    old_text = candidate_report["given_old_text"]
    buggy_method = case["buggy_source"]
    workdir = Path(args.workdir)

    result = {
        "instance_id": case["instance_id"],
        "evaluation_mode": "given_location_full_method",
        "candidate_exactly_inserted": False,
        "outside_method_unchanged": False,
        "test_execution_observed": False,
        "trigger_test_passed": False,
    }

    try:
        validate_method_response(response, buggy_method)
        outside_changes = changes_outside_given_old_text(
            response,
            buggy_method,
            old_text,
        )
    except ValueError as error:
        result.update({
            "evaluation_status": "malformed_method",
            "validation_error": str(error),
        })
        write_result(args.output, result)
        return

    result.update({
        "method_changed": response != buggy_method,
        "changes_outside_given_old_text": outside_changes,
    })

    if args.verify_buggy_baseline:
        baseline = run_defects4j(workdir / "buggy")
        result.update({
            "buggy_compile_returncode": (
                baseline["compile_returncode"]
            ),
            "buggy_compile_output": baseline["compile_output"],
            "buggy_test_output": baseline["test_output"],
            "buggy_test_execution_observed": (
                baseline["test_execution_observed"]
            ),
            "buggy_failing_tests_detected": (
                baseline["failing_tests_detected"]
            ),
        })
        if (
            baseline["compile_returncode"] != 0
            or not baseline["test_execution_observed"]
            or not baseline["failing_tests_detected"]
        ):
            result["evaluation_status"] = "evaluation_error"
            result["validation_error"] = (
                "buggy 基线未满足编译成功且测试失败"
            )
            write_result(args.output, result)
            return

    source_file = workdir / "buggy" / case["source_file"]
    original_bytes = source_file.read_bytes()
    original_text = original_bytes.decode("utf-8")

    try:
        patched_text = replace_exact_method(
            original_text,
            buggy_method,
            response,
        )
    except ValueError as error:
        result.update({
            "evaluation_status": "evaluation_error",
            "validation_error": str(error),
        })
        write_result(args.output, result)
        return

    patched_dir = workdir / args.candidate_dir
    shutil.copytree(workdir / "buggy", patched_dir)
    patched_file = patched_dir / case["source_file"]
    patched_file.write_bytes(patched_text.encode("utf-8"))

    written_text = patched_file.read_bytes().decode("utf-8")
    assert written_text == patched_text
    result.update({
        "method_replacement": "passed",
        "candidate_exactly_inserted": True,
        "outside_method_unchanged": True,
    })

    evaluation = run_defects4j(patched_dir)
    result.update(evaluation)
    result["trigger_test_passed"] = (
        evaluation["developer_tests_passed"]
    )
    result["evaluation_status"] = (
        "trigger_test_passed"
        if evaluation["developer_tests_passed"]
        else "trigger_test_failed"
        if evaluation["compile_returncode"] == 0
        else "compile_failed"
    )
    write_result(args.output, result)


if __name__ == "__main__":
    main()
