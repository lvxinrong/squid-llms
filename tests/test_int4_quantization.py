import torch

from squid.quantize.int4 import (
    dequantize_weight_int4,
    pack_int4,
    quantize_weight_int4,
    unpack_int4,
)


def test_int4_quantization_and_error_bound():
    torch.manual_seed(123)
    weight = torch.randn(8, 16)

    quantized, scale = quantize_weight_int4(weight)
    restored = dequantize_weight_int4(quantized, scale)

    assert quantized.dtype == torch.int8
    assert quantized.min() >= -7
    assert quantized.max() <= 7

    error = (restored - weight).abs()
    tolerance = scale / 2 + 1e-6
    assert (error <= tolerance).all()


def test_int4_pack_and_unpack_known_pair():
    quantized = torch.tensor([[3, -1]], dtype=torch.int8)

    packed = pack_int4(quantized)

    assert packed.dtype == torch.uint8
    assert packed.shape == (1, 1)
    assert packed.item() == 243

    unpacked = unpack_int4(packed, in_features=2)
    assert torch.equal(unpacked, quantized)


def test_int4_pack_and_unpack_odd_columns():
    torch.manual_seed(123)
    weight = torch.randn(3, 5)
    quantized, _ = quantize_weight_int4(weight)

    packed = pack_int4(quantized)

    assert packed.shape == (3, 3)

    unpacked = unpack_int4(packed, in_features=5)
    assert torch.equal(unpacked, quantized)
