"""squid 原生 checkpoint：一次存档，完整还原一个训练现场。

格式约定（自描述）：
    model_state_dict      权重
    config                模型结构（没它权重就是死数据）
    optimizer_state_dict  可选，想续训就要它
    meta                  训练元信息（epoch、val_loss、created_at……）
"""

from datetime import datetime

import torch


def save_checkpoint(path, model, config, optimizer=None, **meta):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "meta": {"created_at": datetime.now().isoformat(timespec="seconds"), **meta},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path):
    """只读文件，不建模型。想干嘛让调用方决定。"""
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(path, device=None):
    """常用组合：读档 → 按 config 建模型 → 载权重 → 上设备。

    权重先落 CPU 再整体搬到目标设备，比直接 map_location=device 稳。
    device=None 时自动探测。
    """
    from squid.device import get_device
    from squid.model.gpt import GPTModel

    checkpoint = load_checkpoint(path)
    model = GPTModel(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device or get_device())
    model.eval()
    return model, checkpoint
