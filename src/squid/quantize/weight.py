import torch

# weight.shape(N, M)
# 量化位置 =（当前权重 / 当前行最大绝对值）× 127
def quantize_weight_int8(weight):
    if weight.ndim != 2:
        raise ValueError("线性层权重必须是二维矩阵")
    if not weight.is_floating_point():
        raise ValueError("量化输入必须是浮点张量")
    # absmax.shape (N, 1)
    absmax = weight.abs().amax(dim=1, keepdim=True)

    scale = absmax / 127
    # torch.where(condition, value_if_true, value_if_false
    scale = torch.where(
        scale == 0,
        torch.ones_like(scale),
        scale,
    )
    # (N, M) / (N, 1)
    # torch.round 四舍五入成整数
    quantized = torch.round(weight / scale) # 取整之后，依然是浮点类型
    # .clamp 限制整数范围 超过上下限的设置为上下限本身 .to(torch.int8) 改变存储类型
    quantized = quantized.clamp(-127, 127).to(torch.int8)

    return quantized, scale

def dequantize_weight_int8(quantized, scale):
    return quantized.to(scale.dtype) * scale