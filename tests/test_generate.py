import torch

from squid.generate.generate_text import generate
from squid.model.gpt import GPTModel
from squid.tokenizer import TiktokenTokenizer

TEST_CFG = {
    "vocab_size": 50257,
    "emb_dim": 64,
    "context_length": 32,
    "n_layers": 2,
    "n_heads": 4,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def test_generate():
    model = GPTModel(TEST_CFG)
    model.eval()
    input_idx = torch.tensor([[15496, 11, 262]])

    # 确定性：temperature=0 时两次生成必须完全一致
    out1 = generate(model, input_idx, 5, 32, temperature=0.0)
    out2 = generate(model, input_idx, 5, 32, temperature=0.0)
    assert torch.equal(out1, out2)

    # 长度：新生成的 token 数不能超过 max_new_tokens
    assert out1.shape[1] <= input_idx.shape[1] + 5

    # 前缀不变：生成的开头必须等于输入
    assert torch.equal(out1[:, :3], input_idx)

    print(TiktokenTokenizer("gpt2").decode(out1[0].tolist()))
    print(TiktokenTokenizer("gpt2").decode(out2[0].tolist()))