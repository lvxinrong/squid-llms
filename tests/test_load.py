import os

import pytest

from squid.checkpoints.gpt2 import load_pretrained_gpt
from squid.config import GPT2_MEDIUM
from squid.generate.text import generate_text
from squid.tokenizer import TiktokenTokenizer

# 竖切终点：加载真实 GPT-2 355M 权重，让 squid 开口说话
GPT2_355M = "/Users/lv/lv_repo/lv_llms/gpt2_355M.pth"


@pytest.mark.skipif(not os.path.exists(GPT2_355M), reason="355M 权重不在本机")
def test_load_355m_and_generate():
    model = load_pretrained_gpt(GPT2_MEDIUM, weights_path=GPT2_355M)
    tokenizer = TiktokenTokenizer("gpt2")

    text = generate_text(
        model, tokenizer, "Every effort moves you",
        max_new_tokens=25, context_size=GPT2_MEDIUM["context_length"],
        temperature=0.0,
    )

    assert text.startswith("Every effort moves you")
    assert len(text) > len("Every effort moves you")
    print("\nsquid 说:", text)
