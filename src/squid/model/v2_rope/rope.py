import torch


def compute_inv_freq(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim: {head_dim} 不是偶数")
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    # 在指数空间里均匀取 head_dim/2 个频率
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    return inv_freq

def compute_cos_sin(
        position_ids: torch.Tensor,
        inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.outer(a, b) 计算两个一维向量的外积，结果是一个二维矩阵
    # out[i][j]=a[i]×b[j]
    freqs = torch.outer(position_ids.float(), inv_freq) # shape (T,) out (head_dim /2) -> (T, head_dim / 2)
    cos = freqs.cos() # (T, head_dim / 2) cos[i][j] = cos(position_ids[i] * inv_freq[j])
    sin = freqs.sin() # (T, head_dim / 2)
    return cos, sin

"""
x.shape (B, H, T, head_dim)
后续attn_score的点积计算 会受到 相对位置 n-m 的影响，而不是不受角度影响。
在不改变 Q/K shape 和向量长度的情况下，对最后一维按固定 pair 注入旋转角度，使后续点积能够感知相对位置。
shape 不变       → 后续矩阵计算可以照常进行
向量长度不变     → 不随意放大或缩小 Q/K
点积获得相对位置 → RoPE 真正要注入的信息
"""
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    # x[..., :half] 和 x[..., half:] 是沿最后一维切前一半 / 后一半
    # 沿最后一维拆成前后两半；x 是当前 Q 或 K，形状为 (B, H, T, head_dim)
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[None, None, :, :].to(device=x.device, dtype=x.dtype)
    sin = sin[None, None, :, :].to(device=x.device, dtype=x.dtype)
    # (B, H, T, head_dim // 2)
    rotated_1 = x1 * cos - x2 * sin
    # (B, H, T, head_dim // 2)
    rotated_2 = x2 * cos + x1 * sin
    # (B, H, T, head_dim)
    return torch.cat([rotated_1, rotated_2], dim=-1)