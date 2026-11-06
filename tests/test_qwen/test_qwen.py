from pathlib import Path

import torch

from squid.checkpoints.qwen import load_qwen2_5
from squid.generate.text import generate_text
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter


def test_qwen_load_and_generate():
    root = Path(__file__).resolve().parents[2]
    weights = root / "data/model_pth/qwen/qwen2_5-0_5b.safetensors"
    model = load_qwen2_5(weights)
    tokenizer = QwenTokenizerAdapter()
    prompt = "你好，介绍一下中国上海的天气情况"
    output = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=200,
        context_size=1024,
        temperature=0.0,
        eos_id=tokenizer.eos_id,
    )
    print(output)
