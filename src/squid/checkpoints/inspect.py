import math
from pathlib import Path
from safetensors import safe_open

def inspect_safetensors(path):
    """只读取 safetensors 元数据，不加载完整权重张量。"""
    path = Path(path)
    tensors = []

    with safe_open(str(path),
                   framework = "pt",
                   device = "cpu") as weights:
        for name in weights.keys():
            tensor_slice = weights.get_slice(name)
            shape = list(tensor_slice.get_shape())

            tensors.append({
                "name": name,
                "shape": shape,
                "dtype": str(tensor_slice.get_dtype()),
                "elements": math.prod(shape)
            })

    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "key_count": len(tensors),
        "total_elements": sum(
            tensor["elements"] for tensor in tensors
        ),
        "tensors": tensors,
    }