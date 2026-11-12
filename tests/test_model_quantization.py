import torch

from squid.model.v3_modern.gpt import GPTModelV3
from squid.quantize.convert import (
    quantize_model_int4,
    quantize_model_int8,
)
from squid.quantize.linear import QuantizedLinearInt8
from squid.quantize.linear_int4 import QuantizedLinearInt4


import torch.nn as nn

TEST_CFG = {
    "vocab_size": 128,
    "emb_dim": 16,
    "ffn_dim": 32,
    "n_heads": 4,
    "n_kv_heads": 2,
    "n_layers": 2,
    "rope_theta": 10000.0,
    "eps": 1e-6,
    "qkv_bias": True,
    "ffn_bias": False,
    "tie_word_embeddings": True,
    "drop_rate": 0.0,
    "context_length": 32,
}


def test_quantize_v3_linear_layers_and_preserve_tied_head():
    torch.manual_seed(123)
    model = GPTModelV3(TEST_CFG).eval()

    replaced = quantize_model_int8(model)

    first_block = model.trf_block[0]
    assert isinstance(first_block.att.W_query, QuantizedLinearInt8)

    assert isinstance(first_block.ffn.down_proj, QuantizedLinearInt8)

    assert len(replaced) == 14
    assert "out_head" not in replaced
    assert isinstance(model.out_head, nn.Linear)

    assert model.out_head.weight is model.tok_emb.weight

    idx = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        logits, past_kv = model(idx, use_cache=True)

    assert logits.shape == (1, 4, 128)
    # torch.isfinite() 用来逐元素检查：这个数是不是有限值（finite）
    # torch.isfinite(logits).all() 整个 logits 张量中不能出现 NaN 或正负无穷。
    assert torch.isfinite(logits).all()
    assert len(past_kv) == 2

    x = torch.tensor([0.0, 1.5, float("inf"), float("-inf"), float("nan")])
    torch.isfinite(x)
    # tensor([True, True, False, False, False])

def test_quantize_v3_int4_layers_and_forward():
    torch.manual_seed(123)
    model = GPTModelV3(TEST_CFG).eval()

    replaced = quantize_model_int4(model)

    first_block = model.trf_block[0]
    assert isinstance(
        first_block.att.W_query,
        QuantizedLinearInt4,
    )
    assert isinstance(
        first_block.ffn.down_proj,
        QuantizedLinearInt4,
    )

    assert len(replaced) == 14
    assert "out_head" not in replaced
    assert model.out_head.weight is model.tok_emb.weight

    idx = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        logits, past_kv = model(idx, use_cache=True)

    assert logits.shape == (1, 4, 128)
    assert torch.isfinite(logits).all()
    assert len(past_kv) == 2
