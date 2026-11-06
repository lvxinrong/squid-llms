from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from squid.checkpoints.qwen import load_qwen2_5
from squid.generate.chat import build_qwen_chat_prompt

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "你好，我叫吕炘嵘。",
    "The capital of France is",
    "def find_even_numbers(numbers):\n    even_numbers = []\n",
]

def test_qwen_logits_compare_with_official():
    device = torch.device("cpu")

    # 只读取本地 Hugging Face 缓存，禁止联网下载
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    history = [
        (
            "你好，我叫吕炘嵘。",
            "你好！很高兴见到你。有什么我可以帮助你的吗？",
        ),
        (
            "我喜欢写代码，也正在学习大模型。",
            "那太好了！编程和大模型都是非常有趣和有挑战性的领域。"
            "你对编程有什么具体的问题或项目吗？",
        ),
    ]

    current_input = (
        "请根据上面的对话回答："
        "用户叫什么名字？用户喜欢做什么？正在学习什么？"
        "不要介绍你自己。"
    )

    prompt = build_qwen_chat_prompt(
        tokenizer,
        history,
        current_input,
    )

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    )

    our_model = load_qwen2_5(
        ROOT / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors",
        device=device,
    )

    official_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    official_model.eval()

    print(
        "official attention:",
        official_model.config._attn_implementation,
    )

    with torch.no_grad():
        our_logits = our_model(input_ids)
        official_logits = official_model(input_ids).logits

        assert our_logits.shape == official_logits.shape
        assert torch.isfinite(our_logits).all()
        assert torch.isfinite(official_logits).all()

        difference = (our_logits - official_logits).abs()
        max_difference = difference.max().item()
        mean_difference = difference.mean().item()

        our_next_token = our_logits[0, -1].argmax().item()
        official_next_token = official_logits[0, -1].argmax().item()

        print("\ninput_ids shape:", tuple(input_ids.shape))
        print("max abs difference:", max_difference)
        print("mean abs difference:", mean_difference)
        print("our next token:", our_next_token, tokenizer.decode([our_next_token]))
        print(
            "official next token:",
            official_next_token,
            tokenizer.decode([official_next_token]),
        )

        torch.testing.assert_close(
            our_logits,
            official_logits,
            rtol=0.0,
            atol=1e-4,
        )
