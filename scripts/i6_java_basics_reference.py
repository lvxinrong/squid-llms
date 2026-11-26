import argparse
import json
import runpy
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = {
    "id",
    "category",
    "signature",
    "specification",
    "public_example",
    "reference_method",
    "public_checks",
    "private_checks",
}

def load_tasks(path):
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(tasks, list) and tasks

    ids = set()
    for task in tasks:
        assert set(task) == REQUIRED_FIELDS, task.get("id")
        assert task["id"] not in ids, task["id"]
        assert task["public_checks"], task["id"]
        assert len(task["private_checks"]) >= 2, task["id"]
        ids.add(task["id"])

    return tasks

def test_source(checks):
    lines = [
        "import java.util.Objects;",
        "",
        "public final class JavaBasicsTest {",
        "    private static int failures = 0;",
        "",
        "    private static void check(",
        "        boolean condition,",
        "        String label",
        "    ) {",
        "        if (!condition) {",
        "            failures++;",
        '            System.out.println("ASSERTION_FAILED: " + label);',
        "        }",
        "    }",
        "",
        "    public static void main(String[] args) {",
    ]

    for index, (kind, expression) in enumerate(checks, start=1):
        lines.append(
            f'        check({expression}, "{kind}-{index}");'
        )

    lines.extend([
        "        if (failures > 0) {",
        '            throw new AssertionError(failures + " assertion(s) failed");',
        "        }",
        '        System.out.println("ALL_ASSERTIONS_PASSED");',
        "    }",
        "}",
        "",
    ])
    return "\n".join(lines)


def run_command(command, cwd):
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "returncode": result.returncode,
        "output": result.stdout + result.stderr,
    }


def evaluate_method(task, method, mode):
    result = {
        "instance_id": task["id"],
        "mode": mode,
        "structure_pass": False,
        "compile_pass": False,
        "behavior_pass": False,
    }

    namespace = runpy.run_path(str(
        ROOT / "scripts/i6_strict_full_method_apply.py"
    ))

    try:
        namespace["validate_method_response"](
            method,
            task["signature"] + " {",
        )
    except ValueError as error:
        result.update({
            "status": "malformed_method",
            "error": str(error),
        })
        return result

    result["structure_pass"] = True
    checks = [
        ("public", expression)
        for expression in task["public_checks"]
    ] + [
        ("private", expression)
        for expression in task["private_checks"]
    ]

    with tempfile.TemporaryDirectory(
        prefix=f"i6-j0-{task['id'].lower()}-"
    ) as temporary:
        workdir = Path(temporary)
        (workdir / "JavaBasics.java").write_text(
            "public final class JavaBasics {\n"
            + task["reference_method"]
            + "\n}\n",
            encoding="utf-8",
        )
        (workdir / "JavaBasicsTest.java").write_text(
            test_source(checks),
            encoding="utf-8",
        )
        (workdir / "classes").mkdir()

        compile_result = run_command(
            [
                "javac",
                "-encoding",
                "UTF-8",
                "-d",
                "classes",
                "JavaBasics.java",
                "JavaBasicsTest.java",
            ],
            workdir,
        )
        result["compile_output"] = compile_result["output"]
        result["compile_returncode"] = compile_result["returncode"]

        if compile_result["returncode"] != 0:
            result["status"] = "compile_failed"
            return result

        result["compile_pass"] = True
        test_result = run_command(
            ["java", "-ea", "-cp", "classes", "JavaBasicsTest"],
            workdir,
        )
        result["test_output"] = test_result["output"]
        result["test_returncode"] = test_result["returncode"]

        if (
            test_result["returncode"] == 0
            and "ALL_ASSERTIONS_PASSED" in test_result["output"]
        ):
            result.update({
                "behavior_pass": True,
                "status": "passed",
            })
        else:
            result["status"] = "assertion_failed"

    return result

def evaluate_reference(task):
    return evaluate_method(
        task,
        task["reference_method"],
        "reference",
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        result = evaluate_reference(task)
        results.append(result)

        report_path = report_dir / f"{task['id'].lower()}.json"
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(task["id"], result["status"], f"report: {report_path}")

    passed = sum(
        result["status"] == "passed"
        for result in results
    )
    print(f"reference controls: {passed}/{len(results)} passed")

    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()