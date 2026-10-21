import torch

from squid.model.gpt import GPTModel

# 测试用小配置：跑得动就行，不追求模型能力
TEST_CFG = {
    "vocab_size": 50257,
    "emb_dim": 64,
    "context_length": 32,
    "n_layers": 2,
    "n_heads": 4,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def test_gpt_model_output_shape():
    """前向输出形状必须是 (batch, seq_len, vocab_size)。"""
    model = GPTModel(TEST_CFG)
    x = torch.randint(0, 50257, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, 50257)


def test_gpt_model_no_nan():
    """输出里不能有 NaN。"""
    model = GPTModel(TEST_CFG)
    x = torch.randint(0, 50257, (2, 8))
    logits = model(x)
    assert not torch.isnan(logits).any()


def test_gpt_model_respects_context_length():
    """超过 context_length 的输入必须报错，不能静默出错。"""
    model = GPTModel(TEST_CFG)
    x = torch.randint(0, 50257, (1, TEST_CFG["context_length"] + 1))
    try:
        model(x)
        assert False, "超长输入应该报错"
    except (IndexError, RuntimeError):
        pass
