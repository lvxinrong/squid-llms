import torch
from safetensors.torch import save_file

from squid.checkpoints.inspect import inspect_safetensors


def test_inspect_safetensors_reads_metadata(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "weight": torch.zeros(2, 3, dtype=torch.float32),
            "bias": torch.zeros(3, dtype=torch.bfloat16),
        },
        str(path),
    )

    result = inspect_safetensors(path)

    assert result["key_count"] == 2
    assert result["total_elements"] == 9
    assert result["file_bytes"] > 0

    by_name = {
        tensor["name"]: tensor
        for tensor in result["tensors"]
    }

    assert by_name["weight"]["shape"] == [2, 3]
    assert by_name["weight"]["elements"] == 6
    assert by_name["bias"]["shape"] == [3]
    assert by_name["bias"]["elements"] == 3
