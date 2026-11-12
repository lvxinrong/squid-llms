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


def bytes_to_mb(value):
    return round(value / 1_000_000, 2)


def memory_snapshot(device):
    if device.type == "mps":
        current = torch.mps.current_allocated_memory()
        driver = torch.mps.driver_allocated_memory()
        return {
            "current_allocated_bytes": current,
            "current_allocated_mb": bytes_to_mb(current),
            "driver_allocated_bytes": driver,
            "driver_allocated_mb": bytes_to_mb(driver),
        }

    if device.type == "cuda":
        current = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak_reserved = torch.cuda.max_memory_reserved()
        return {
            "current_allocated_bytes": current,
            "current_allocated_mb": bytes_to_mb(current),
            "peak_allocated_bytes": peak,
            "peak_allocated_mb": bytes_to_mb(peak),
            "current_reserved_bytes": reserved,
            "current_reserved_mb": bytes_to_mb(reserved),
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_mb": bytes_to_mb(peak_reserved),
        }

    return {}
