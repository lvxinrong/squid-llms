"""M3 验收（真实权重）：355M 下，cache 与朴素生成输出一致 + 提速可测。"""
import time

import torch

from squid.checkpoints.gpt2 import load_pretrained_gpt
from squid.device import get_device
from squid.generate.engine import generate, generate_no_cache
from squid.tokenizer import TiktokenTokenizer

CFG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def test_kv_cache_with_real_weights():
    device = get_device()
    model = load_pretrained_gpt(CFG, "/Users/lv/lv_repo/lv_llms/gpt2_355M.pth", device)
    model.eval()

    tokenizer = TiktokenTokenizer("gpt2")
    start_context = "The future of human and artificial intelligence is"
    idx = torch.tensor([tokenizer.encode(start_context)], dtype=torch.long).to(device)

    t0 = time.perf_counter()
    torch.manual_seed(123)
    out_cache = generate(model, idx.clone(), max_new_tokens=512, context_size=1024, temperature=0.8, top_k=40)
    t_cache = time.perf_counter() - t0

    t0 = time.perf_counter()
    torch.manual_seed(123)
    out_naive = generate_no_cache(model, idx.clone(), max_new_tokens=512, context_size=1024, temperature=0.8, top_k=40)
    t_naive = time.perf_counter() - t0

    assert torch.equal(out_cache, out_naive), "cache 与朴素版输出必须逐字一致"
    print(f"\ncache 版 {t_cache:.2f}s | 朴素版 {t_naive:.2f}s | 提速 {t_naive/t_cache:.1f}x")
    english = tokenizer.decode(out_cache[0].tolist())
    # print("\n=== 英文生成 ===\n", english)

    from squid.eval.judge import get_client
    resp = get_client().chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": f"把下面的英文翻译成自然流畅的中文，保留气质，适合发朋友圈：\n\n{english}"}],
        temperature=0.7,
    )
    print("\n=== 中文翻译 ===\n", resp.choices[0].message.content)
