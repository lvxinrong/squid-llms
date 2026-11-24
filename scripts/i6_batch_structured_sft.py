import argparse
import json
import runpy
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--exclude", default="Lang-1,Lang-3")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    discovery_namespace = runpy.run_path(str(
        root / "scripts/i6_build_sft_data.py"
    ))
    discover_case_paths = discovery_namespace["discover_case_paths"]

    train_paths, val_paths, skipped = discover_case_paths(
        args.case_dir
    )
    excluded = set(args.exclude.split(",")) - {""}

    train_entries = []
    val_entries = []
    failures = []

    with tempfile.TemporaryDirectory(prefix="i6-structured-parts-") as tmp:
        tmp_dir = Path(tmp)

        for data_role, paths, destination in (
            ("train", train_paths, train_entries),
            ("validation", val_paths, val_entries),
        ):
            for case_path in paths:
                case = json.loads(
                    case_path.read_text(encoding="utf-8")
                )
                instance_id = case["instance_id"]

                if instance_id in excluded:
                    continue

                key_path = case_path.with_name(
                    case_path.name.replace("-case.json", "-key.json")
                )
                part_path = tmp_dir / f"{instance_id}.json"

                result = subprocess.run(
                    [
                        sys.executable,
                        str(root / "scripts/i6_structured_sft_data.py"),
                        "--case",
                        str(case_path),
                        "--key",
                        str(key_path),
                        "--output",
                        str(part_path),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    failures.append({
                        "instance_id": instance_id,
                        "data_role": data_role,
                        "error": result.stderr[-2000:],
                    })
                    continue

                entry = json.loads(
                    part_path.read_text(encoding="utf-8")
                )[0]
                destination.append(entry)

    for output, entries in (
        (Path(args.train_output), train_entries),
        (Path(args.val_output), val_entries),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"output: {output}，samples={len(entries)}")

    print(f"skipped legacy case files: {len(skipped)}")
    print(f"conversion failures: {len(failures)}")
    print("boundary: 仅构造结构化训练数据，未训练模型，不证明泛化")

    if failures:
        print(json.dumps(failures, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
