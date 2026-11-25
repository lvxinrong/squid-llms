import json
import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i6_given_location_baseline.py"
))


def test_only_actual_trailing_eos_is_removed():
    generated = [10, 20, 99]

    content, finish = namespace["split_generated_ids"](
        generated,
        eos_id=99,
    )

    assert content == [10, 20]
    assert finish == "stop"
    assert generated == [10, 20, 99]


def test_missing_eos_is_length_limited():
    content, finish = namespace["split_generated_ids"](
        [10, 20],
        eos_id=99,
    )

    assert content == [10, 20]
    assert finish == "length"


def test_json_roundtrip_preserves_exact_new_text():
    prepared = {
        "target_method": "demo",
        "old_text": "old",
    }
    new_text = "\n    regex.append('\\\\');\n"

    edit = namespace["make_edit"](prepared, new_text)
    restored = json.loads(
        json.dumps(edit, ensure_ascii=False)
    )

    assert restored["new_text"] == new_text
