import argparse
import json
import re
import runpy
import subprocess
from collections import Counter
from pathlib import Path


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+")
METHOD_RE = re.compile(
    r"^\s*(?:public|protected|private)\s+"
    r"(?:static\s+)?(?:final\s+)?"
    r"[\w$<>\[\],.?]+\s+(\w+)\s*\("
)
CONSTRUCTOR_RE = re.compile(
    r"^\s*(?:public|protected|private)\s+"
    r"(?:final\s+)?([A-Z]\w*)\s*\("
)


def parse_hunks(diff_text):
    hunks = []

    for line in diff_text.splitlines():
        match = HUNK_RE.match(line)
        if match:
            hunks.append({
                "old_start": int(match.group(1)),
                "old_count": (
                    int(match.group(2))
                    if match.group(2) is not None
                    else 1
                ),
            })

    return hunks


def method_declarations(lines):
    declarations = []

    for index, line in enumerate(lines):
        method_match = METHOD_RE.match(line)
        constructor_match = CONSTRUCTOR_RE.match(line)

        if method_match:
            declarations.append({
                "index": index,
                "name": method_match.group(1),
                "kind": "method",
            })
        elif constructor_match:
            declarations.append({
                "index": index,
                "name": constructor_match.group(1),
                "kind": "constructor",
            })

    return declarations


def locate_methods(lines, hunks):
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(
        root / "scripts/i6_build_case.py"
    ))
    extract_java_block = namespace["extract_java_block"]

    declarations = method_declarations(lines)
    changed = set()

    for hunk in hunks:
        line_number = hunk["old_start"]
        found = False

        for declaration in declarations:
            start = declaration["index"]
            if start + 1 > line_number:
                continue

            marker = lines[start].strip()
            try:
                block = extract_java_block(lines, marker)
            except (ValueError, StopIteration):
                continue

            if (
                block["start_line"] <= line_number
                <= block["end_line"]
            ):
                changed.add(
                    f"{declaration['kind']}:{declaration['name']}"
                )
                found = True
                break

        if not found:
            changed.add("class_level_or_unknown")

    return sorted(changed)


def audit_file(buggy_file, fixed_file):
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

    diff_text = result.stdout
    hunks = parse_hunks(diff_text)
    lines = buggy_file.read_text(
        encoding="utf-8"
    ).splitlines()

    changed_methods = locate_methods(lines, hunks)

    if not diff_text:
        classification = "no_source_difference"
    elif not hunks:
        classification = "invalid_or_unparsed_diff"
    elif changed_methods == ["class_level_or_unknown"]:
        classification = "class_level_only"
    elif "class_level_or_unknown" in changed_methods:
        classification = "contains_class_level_change"
    elif len(changed_methods) == 1:
        classification = "single_method_candidate"
    else:
        classification = "multi_method"

    return {
        "file": buggy_file.name,
        "hunks": hunks,
        "changed_methods": changed_methods,
        "classification": classification,
        "diff": diff_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    result = audit_file(
        workdir / "buggy" / args.source_file,
        workdir / "fixed" / args.source_file,
    )
    result["workdir"] = str(workdir)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        key: value
        for key, value in result.items()
        if key != "diff"
    }, ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
