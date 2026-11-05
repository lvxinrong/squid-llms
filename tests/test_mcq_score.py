import json
import random
from pathlib import Path

from squid.checkpoints.gpt2 import load_pretrained_gpt
from squid.checkpoints.store import load_model
from squid.device import get_device
from squid.eval.mcq import evaluate_answer_likelihood, evaluate_mcq
from squid.tokenizer import TiktokenTokenizer


def load_gpt_model(cfg, tokenizer, device):
    base_path = Path("/Users/lv/lv_repo/squid-llms/data/model_pth/gpt2/gpt2_355M.pth")
    baseModel = load_pretrained_gpt(cfg, base_path, device)
    baseModel.eval()
    sft_path = Path("/Users/lv/lv_repo/squid-llms/data/model_pth/gpt2/gpt2_355M_instruct.pth")
    sftModel, _ = load_model(sft_path, device)
    sftModel.eval()

    return baseModel, sftModel

def test_mcq_score():
    ROOT_PATH = Path("/Users/lv/lv_repo/squid-llms/data/mcq")
    tokenizer = TiktokenTokenizer("gpt2")
    cfg = {
        "vocab_size": tokenizer.tok.n_vocab,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.05,
        "qkv_bias": True
    }
    device = get_device()
    baseModel, sftModel = load_gpt_model(cfg, tokenizer, device)
    data_path = ROOT_PATH / "test_data.json"
    base_test_data = json.loads(data_path.read_text(encoding="utf-8"))
    sft_test_data = json.loads(data_path.read_text(encoding="utf-8"))
    base_response_path = ROOT_PATH / "report-base.json"
    sft_response_path = ROOT_PATH / "report-sft.json"
    base_response = evaluate_mcq(baseModel, tokenizer, base_test_data, base_response_path)
    sft_response = evaluate_mcq(sftModel, tokenizer, sft_test_data, sft_response_path)
    print(base_response)
    print(sft_response)


def shuffle_mcq_data(test_data, seed=123):
    rng = random.Random(seed)
    shuffled_data = []

    for entry in test_data:
        options = entry["options"].copy()
        correct_option = options[entry["answer"]]
        rng.shuffle(options)

        shuffled_data.append({
            **entry,
            "options": options,
            "answer": options.index(correct_option),
        })

    return shuffled_data


def test_mcq_score_shuffled():
    root_path = Path("/Users/lv/lv_repo/squid-llms/data/mcq")
    tokenizer = TiktokenTokenizer("gpt2")
    cfg = {
        "vocab_size": tokenizer.tok.n_vocab,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.05,
        "qkv_bias": True
    }
    device = get_device()
    base_model, sft_model = load_gpt_model(cfg, tokenizer, device)

    data_path = root_path / "test_data.json"
    base_test_data = shuffle_mcq_data(
        json.loads(data_path.read_text(encoding="utf-8"))
    )
    sft_test_data = shuffle_mcq_data(
        json.loads(data_path.read_text(encoding="utf-8"))
    )

    base_response = evaluate_mcq(
        base_model,
        tokenizer,
        base_test_data,
        root_path / "report-base-shuffled.json",
    )
    sft_response = evaluate_mcq(
        sft_model,
        tokenizer,
        sft_test_data,
        root_path / "report-sft-shuffled.json",
    )

    print("base shuffled:", base_response["summary"])
    print("SFT shuffled:", sft_response["summary"])


def test_mcq_answer_likelihood():
    root_path = Path("/Users/lv/lv_repo/squid-llms/data/mcq")
    tokenizer = TiktokenTokenizer("gpt2")
    cfg = {
        "vocab_size": tokenizer.tok.n_vocab,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.05,
        "qkv_bias": True
    }
    device = get_device()
    base_model, sft_model = load_gpt_model(cfg, tokenizer, device)

    data_path = root_path / "test_data.json"
    base_test_data = json.loads(data_path.read_text(encoding="utf-8"))
    sft_test_data = json.loads(data_path.read_text(encoding="utf-8"))

    base_response = evaluate_answer_likelihood(
        base_model,
        tokenizer,
        base_test_data,
        root_path / "report-base-answer-likelihood.json",
    )
    sft_response = evaluate_answer_likelihood(
        sft_model,
        tokenizer,
        sft_test_data,
        root_path / "report-sft-answer-likelihood.json",
    )

    print("base answer likelihood:", base_response["summary"])
    print("SFT answer likelihood:", sft_response["summary"])
