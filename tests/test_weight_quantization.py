import torch
from torch import nn

from squid.quantize.linear import QuantizedLinearInt8
from squid.quantize.weight import (
    dequantize_weight_int8,
    quantize_weight_int8,
)
from torch.nn import functional as F



def test_int8_quantization_is_per_row_and_handles_zero():
    weight = torch.tensor([
        [0.0, 0.5, -1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    quantized, scale = quantize_weight_int8(weight)

    assert quantized.dtype == torch.int8
    assert quantized.shape == weight.shape
    assert scale.shape == (2, 1)

    assert torch.equal(
        quantized[0],
        torch.tensor([0, 64, -127, 127], dtype=torch.int8),
    )
    assert torch.equal(
        quantized[1],
        torch.zeros(4, dtype=torch.int8),
    )

    assert torch.isfinite(scale).all()
    assert scale[1, 0].item() == 1.0

    restored = dequantize_weight_int8(quantized, scale)
    assert torch.equal(restored[1], weight[1])


def test_int8_round_trip_error_is_bounded_by_half_scale():
    torch.manual_seed(123)
    weight = torch.randn(64, 32)

    quantized, scale = quantize_weight_int8(weight)
    restored = dequantize_weight_int8(quantized, scale)

    error = (restored - weight).abs()
    tolerance = scale / 2 + 1e-6

    assert (error <= tolerance).all()

