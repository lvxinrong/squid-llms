import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from squid.checkpoints.gpt2 import load_pretrained_gpt
from squid.checkpoints.store import load_model
from squid.data.instruct import format_input
from squid.device import get_device
from squid.generate.engine import generate, generate_no_cache
from squid.eval.judge import score_entries, summarize_scores
from squid.tokenizer import TiktokenTokenizer

def generate_test_data_response(model, test_data, tokenizer, save_path, device, use_cache=True):
    for i, entry in tqdm(enumerate(test_data)):
        input_text = format_input(entry) + "\n\n### Response:\n"
        idx = tokenizer.encode(input_text, allow_special=True)
        idx = torch.tensor([idx], dtype=torch.long).to(device)
        if use_cache:
            token_ids = generate(model, idx, 256, 1024, eos_id=tokenizer.eos_id)
        else:
            token_ids = generate_no_cache(model, idx, 256, 1024, eos_id=tokenizer.eos_id)
        generated_text = tokenizer.decode(token_ids[0].tolist())
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        response_text = response_text.replace("<|endoftext|>", "").strip()
        test_data[i]["model_response"] = response_text
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

def test_compare_base_sft_testdata():
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
    base_path = Path("/Users/lv/lv_repo/squid-llms/data/model_pth/gpt2_355M.pth")
    baseModel = load_pretrained_gpt(cfg, base_path, device)
    baseModel.eval()
    sft_path = Path("/Users/lv/lv_repo/squid-llms/data/model_pth/gpt2_355M_instruct.pth")
    sftModel, _ = load_model(sft_path, device)
    sftModel.eval()
    instruct_train_data_path = Path("/Users/lv/lv_repo/squid-llms/data/instruct/instruction-data.json")
    data = json.loads(instruct_train_data_path.read_text(encoding="utf-8"))
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    test_data = data[train_portion : train_portion + test_portion]
    test_data = test_data[:5]
    test_data = test_data[:10]  # 先小批量跑通评委链路
    base_out = Path("/Users/lv/lv_repo/squid-llms/data/instruct/instruction-data-response-base.json")
    sft_out = Path("/Users/lv/lv_repo/squid-llms/data/instruct/instruction-data-response-sft.json")

    torch.manual_seed(123)
    generate_test_data_response(baseModel, test_data, tokenizer, base_out, device)
    torch.manual_seed(123)
    generate_test_data_response(sftModel, test_data, tokenizer, sft_out, device)

    print("\n=== base 模型评委打分 ===")
    base_scored = score_entries(json.loads(base_out.read_text(encoding="utf-8")), save_path=str(base_out))
    summarize_scores(base_scored)

    print("\n=== SFT 模型评委打分 ===")
    sft_scored = score_entries(json.loads(sft_out.read_text(encoding="utf-8")), save_path=str(sft_out))
    summarize_scores(sft_scored)


def test_sft_generate_kv_cache():
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
    sft_path = Path("/Users/lv/lv_repo/squid-llms/data/model_pth/gpt2_355M_instruct.pth")
    sftModel, _ = load_model(sft_path, device)
    sftModel.eval()
    instruct_train_data_path = Path("/Users/lv/lv_repo/squid-llms/data/instruct/instruction-data.json")
    data = json.loads(instruct_train_data_path.read_text(encoding="utf-8"))
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    test_data = data[train_portion: train_portion + test_portion]

    sft_out_cache = Path("/Users/lv/lv_repo/squid-llms/data/instruct/instruction-data-response-sft.json")
    torch.manual_seed(123)
    t0 = time.perf_counter()
    generate_test_data_response(sftModel, test_data, tokenizer, sft_out_cache, device, True)
    t_cache = time.perf_counter() - t0
    print("\n=== SFT 模型评委打分 ===")
    sft_scored = score_entries(json.loads(sft_out_cache.read_text(encoding="utf-8")), save_path=str(sft_out_cache))
    summarize_scores(sft_scored)

    sft_out_nocache = Path("/Users/lv/lv_repo/squid-llms/data/instruct/instruction-data-response-sft-nocache.json")
    torch.manual_seed(123)
    t0 = time.perf_counter()
    generate_test_data_response(sftModel, test_data, tokenizer, sft_out_nocache, device, False)
    t_naive = time.perf_counter() - t0
    print("\n=== SFT 模型评委打分 ===")
    sft_scored = score_entries(json.loads(sft_out_nocache.read_text(encoding="utf-8")), save_path=str(sft_out_nocache))
    summarize_scores(sft_scored)

    cache_resp = json.loads(sft_out_cache.read_text())
    naive_resp = json.loads(sft_out_nocache.read_text())
    assert [e["model_response"] for e in cache_resp] == [e["model_response"] for e in
                                                         naive_resp], "cache 与 no_cache 输出必须一致"
    print(f"\ncache 版 {t_cache:.2f}s | 朴素版 {t_naive:.2f}s | 提速 {t_naive / t_cache:.1f}x")


