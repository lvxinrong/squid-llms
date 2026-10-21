import math

import torch
from torch import nn


class LoRALayer(nn.Module):
    """低秩旁路：ΔW = A @ B，B 零初始化保证训练起点 = 原模型行为。"""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.rank = rank
        self.alpha = alpha

    def forward(self, x):
        return (self.alpha / self.rank) * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """Linear + LoRA 并联：y = Wx + (α/r)·BAx。原 Linear 冻结，只训旁路。"""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    def merged_weight(self) -> torch.Tensor:
        """把旁路折回主权重：W' = W + (α/r)·(A@B).T"""
        return self.linear.weight + (self.lora.alpha / self.lora.rank) * (self.lora.A @ self.lora.B).T


def replace_linear_with_lora(model: nn.Module, rank: int, alpha: float):
    for name, module in model.named_children():
        if isinstance(module, LinearWithLoRA):
            continue
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)
