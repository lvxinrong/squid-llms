import pytest
import torch
from squid.generate.engine import generate
from squid.model.v1_gpt.gpt import GPTModel
from squid.tokenizer import TiktokenTokenizer
from squid.model.v2_rope.gpt import GPTModelRoPE

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

@pytest.mark.parametrize(
    "model_cls",
    [GPTModel, GPTModelRoPE],
)
def test_generate_with_past_kv_matches_full_recompute(model_cls):
    torch.manual_seed(123)
    model = model_cls(TEST_CFG)
    model.eval()

    first_prompt = torch.tensor([[15496, 11, 262]])

    first_out, past_kv = generate(
        model, first_prompt, max_new_tokens = 3, context_size = 32, temperature = 0.0, return_past_kv = True,
    )

    continuation = torch.tensor([[333, 777]])

    full_prompt = torch.cat([first_out, continuation], dim = 1)

    expected, expected_past = generate(
        model,
        full_prompt,
        max_new_tokens = 3,
        context_size = 32,
        temperature = 0.0,
        return_past_kv = True,
    )

    actual, actual_past = generate(
        model,
        continuation,
        max_new_tokens=3,
        context_size=32,
        temperature=0.0,
        past_kv=past_kv,
        return_past_kv=True,
    )

    assert torch.equal(
        actual[:, :continuation.shape[1]],
        continuation,
    )

    assert torch.equal(
        actual[:, continuation.shape[1]:],
        expected[:, full_prompt.shape[1]:],
    )

    assert (actual_past[0][0].shape[2] == expected_past[0][0].shape[2])
