import torch
from torch import nn
from torch.nn import functional as F

from squid.quantize.linear import QuantizedLinearInt8
from squid.quantize.weight import dequantize_weight_int8


def test_quantized_linear_forward_and_storage():
    linear = nn.Linear(4, 3, bias=True)

    with torch.no_grad():
        linear.weight.copy_(torch.tensor([
            [0.0, 0.5, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, -2.0],
        ]))
        linear.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))

    quantized_linear = QuantizedLinearInt8.from_float(linear)
    x = torch.tensor([[1.0, 2.0, -1.0, 0.5]])

    actual = quantized_linear(x)

    dequantized_weight = dequantize_weight_int8(
        quantized_linear.quantized_weight,
        quantized_linear.weight_scale,
    )
    expected = F.linear(
        x,
        dequantized_weight,
        quantized_linear.bias,
    )

    assert torch.equal(actual, expected)
    assert torch.allclose(
        actual,
        linear(x),
        rtol=0,
        atol=0.02,
    )

    assert quantized_linear.quantized_weight.dtype == torch.int8
    assert quantized_linear.weight_scale.shape == (3, 1)
    assert not quantized_linear.quantized_weight.requires_grad


def test_quantized_linear_preserves_none_bias():
    linear = nn.Linear(4, 3, bias=False)
    quantized_linear = QuantizedLinearInt8.from_float(linear)

    assert quantized_linear.bias is None

    x = torch.randn(2, 4)
    actual = quantized_linear(x)

    dequantized_weight = dequantize_weight_int8(
        quantized_linear.quantized_weight,
        quantized_linear.weight_scale,
    )
    expected = F.linear(x, dequantized_weight, None)

    assert torch.equal(actual, expected)