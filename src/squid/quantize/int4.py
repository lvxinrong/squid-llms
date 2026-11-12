import torch


def quantize_weight_int4(weight):
    if weight.ndim != 2:
        raise ValueError("线性层权重必须是二维矩阵")
    if not weight.is_floating_point():
        raise ValueError("量化输入必须是浮点张量")

    absmax = weight.abs().amax(dim=1, keepdim=True)
    scale = absmax / 7
    scale = torch.where(
        scale == 0,
        torch.ones_like(scale),
        scale,
    )

    quantized = torch.round(weight / scale)
    quantized = quantized.clamp(-7, 7).to(torch.int8)

    return quantized, scale


def pack_int4(quantized):
    if quantized.dtype != torch.int8:
        raise TypeError("pack_int4 输入必须是 int8")

    encoded = (
        quantized.to(torch.int16) & 0x0F
    ).to(torch.uint8)

    if encoded.shape[1] % 2 != 0:
        padding = torch.zeros(
            encoded.shape[0],
            1,
            dtype=torch.uint8,
            device=encoded.device,
        )
        encoded = torch.cat([encoded, padding], dim=1)

    low = encoded[:, 0::2]
    high = encoded[:, 1::2]

    return low | (high << 4)

# in_features 原始权重数量 packed 压缩后的权重
def unpack_int4(packed, in_features):
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    unsigned = torch.stack(
        [low, high],
        dim=-1,
    ).reshape(packed.shape[0], -1)

    unsigned = unsigned[:, :in_features]
    signed = unsigned.to(torch.int16)
    signed = torch.where(
        signed >= 8,
        signed - 16,
        signed,
    )

    return signed.to(torch.int8)


def dequantize_weight_int4(quantized, scale):
    return quantized.to(scale.dtype) * scale
