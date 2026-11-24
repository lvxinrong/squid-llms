import json
import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_build_sft_data.py"
))
discover_case_paths = namespace["discover_case_paths"]


def test_discover_case_paths_uses_roles_and_skips_legacy_holdout(tmp_path):
    fixtures = [
        ("train-case.json", "train_candidate", False),
        ("val-case.json", "validation_candidate", True),
    ]

    for name, role, excluded in fixtures:
        (tmp_path / name).write_text(
            json.dumps({
                "data_role": role,
                "excluded_from_training": excluded,
            }),
            encoding="utf-8",
        )

    (tmp_path / "legacy-case.json").write_text(
        json.dumps({"instance_id": "Lang-1"}),
        encoding="utf-8",
    )

    train, val, skipped = discover_case_paths(tmp_path)

    assert [path.name for path in train] == ["train-case.json"]
    assert [path.name for path in val] == ["val-case.json"]
    assert [path.name for path in skipped] == ["legacy-case.json"]
