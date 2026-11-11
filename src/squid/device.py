import torch


def get_device() -> torch.device:
    """唯一的 device 入口。CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reset_peak_memory_stats(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def memory_snapshot(device):
    if device.type == "mps":
        return {
            "current_allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        }

    if device.type == "cuda":
        return {
            "current_allocated_bytes": torch.cuda.memory_allocated(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "current_reserved_bytes": torch.cuda.memory_reserved(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    return {}
