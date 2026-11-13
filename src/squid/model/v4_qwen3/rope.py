import torch

def compute_inv_freq(head_dim, base=10000.0):
    if head_dim % 2 != 0:
        raise ValueError("head_dim 必须是偶数")

    channel_range = torch.arange(
        0,
        head_dim,
        2,
        dtype=torch.float32,
    )
    return 1.0 / (base ** (channel_range / head_dim))

def compute_cos_sin(position_ids, inv_freq):
    freqs = torch.outer(
        position_ids.float(),
        inv_freq,
    )
    return freqs.cos(), freqs.sin()

# x.shape (B, H_heads, T, head_dim)
def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # 在 batch 和 head 两个维度上补出大小为 1 的轴，让同一份 cos / sin 能广播到所有 batch 和所有 head。
    # cos.shape (1, 1, T, head_dim / 2)
    # cos = cos[None, None, :, :] 等价于 cos.unsqueeze(0).unsqueeze(0)
    cos = cos[None, None, :, :].to(device=x.device, dtype=x.dtype)
    sin = sin[None, None, :, :].to(device=x.device, dtype=x.dtype)

    rotated_1 = x1 * cos - x2 * sin
    rotated_2 = x2 * cos + x1 * sin
    return torch.cat([rotated_1, rotated_2], dim = -1)