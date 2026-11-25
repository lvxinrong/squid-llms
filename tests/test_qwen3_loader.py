import json
import runpy
from pathlib import Path


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "src/squid/checkpoints/qwen3.py"
))


def test_sharded_qwen3_weights_are_merged(
    tmp_path,
    monkeypatch,
):
    index = {
        "weight_map": {
            "weight.a": "model-00001-of-00002.safetensors",
            "weight.b": "model-00002-of-00002.safetensors",
            "weight.c": "model-00001-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )

    def fake_load_file(path, device):
        name = Path(path).name
        return {name: device}

    monkeypatch.setitem(
        namespace["load_qwen3_weights"].__globals__,
        "load_file",
        fake_load_file,
    )

    result = namespace["load_qwen3_weights"](tmp_path)

    assert len(result) == 2
    assert result["model-00001-of-00002.safetensors"] == "cpu"
    assert result["model-00002-of-00002.safetensors"] == "cpu"
