import torch
from torch import nn
from torch.nn import functional as F

from squid.quantize.int4 import (
    dequantize_weight_int4,
    unpack_int4,
)
from squid.quantize.linear_int4 import QuantizedLinearInt4


def test_quantized_linear_int4_forward_and_packed_storage():
    torch.manual_seed(123)
    linear = nn.Linear(5, 3, bias=True)

    quantized_linear = QuantizedLinearInt4.from_float(linear)

    assert quantized_linear.packed_weight.dtype == torch.uint8
    assert quantized_linear.packed_weight.shape == (3, 3)
    assert quantized_linear.in_features == 5
    assert not quantized_linear.packed_weight.requires_grad

    x = torch.randn(2, 5)
    actual = quantized_linear(x)

    unpacked = unpack_int4(
        quantized_linear.packed_weight,
        in_features=5,
    )
    weight = dequantize_weight_int4(
        unpacked,
        quantized_linear.weight_scale,
    )
    expected = F.linear(
        x,
        weight,
        quantized_linear.bias,
    )

    assert torch.equal(actual, expected)
