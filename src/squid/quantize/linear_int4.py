import torch
from torch import nn
from torch.nn import functional as F

from squid.quantize.int4 import (
    dequantize_weight_int4,
    pack_int4,
    quantize_weight_int4,
    unpack_int4,
)

class QuantizedLinearInt4(nn.Module):
    def __init__(self, packed_weight, weight_scale, in_features, bias = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = packed_weight.shape[0]

        self.register_buffer(
            "packed_weight",
            packed_weight,
        )
        self.register_buffer(
            "weight_scale",
            weight_scale,
        )

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear):
        quantized_weight, weight_scale = quantize_weight_int4(
            linear.weight.detach(),
        )
        packed_weight = pack_int4(quantized_weight)

        bias = (
            linear.bias.detach().clone()
            if linear.bias is not None
            else None
        )

        return cls(
            packed_weight,
            weight_scale,
            linear.in_features,
            bias,
        )

    def forward(self, x):
        quantized_weight = unpack_int4(
            self.packed_weight,
            self.in_features,
        )
        weight = dequantize_weight_int4(
            quantized_weight,
            self.weight_scale,
        )
        return F.linear(x, weight, self.bias)