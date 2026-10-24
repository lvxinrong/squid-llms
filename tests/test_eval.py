import torch

from squid.eval.generate import generate_responses
from squid.eval.judge import parse_score, summarize_scores
from squid.model.gpt import GPTModel
from squid.tokenizer import TiktokenTokenizer


def test_parse_score():
    assert parse_score("85") == 85
    assert parse_score("I'd give it 72.") == 72
    assert parse_score("  100\n") == 100
    assert parse_score("abc") is None
    assert parse_score("") is None
    assert parse_score("150") == 100   # 越界截断
    assert parse_score("-5") == 0


def test_summarize_empty(capsys):
    summarize_scores([{"score": None}])
    assert "没有有效打分" in capsys.readouterr().out


def test_generate_responses(tmp_path):
    cfg = {"vocab_size": 50257, "emb_dim": 32, "context_length": 64,
           "n_layers": 1, "n_heads": 2, "drop_rate": 0.0, "qkv_bias": False}
    model = GPTModel(cfg)
    tokenizer = TiktokenTokenizer("gpt2")
    data = [{"instruction": "Say hi.", "input": "", "output": "Hi"}]

    out_path = tmp_path / "responses.json"
    responses = generate_responses(
        model, tokenizer, data, out_path,
        max_new_tokens=3, context_size=64,
    )

    assert len(responses) == 1
    assert "model_response" in responses[0]
    # response 不含 prompt 和模板残留
    assert "### Response:" not in responses[0]["model_response"]
    assert "Say hi." not in responses[0]["model_response"]
    assert out_path.exists()
