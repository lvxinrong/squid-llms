from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from squid.checkpoints.qwen import load_qwen2_5


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

    our_model = load_qwen2_5(
        ROOT / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors",
        device=device,
    )

    official_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    official_model.eval()

    with torch.no_grad():
        for prompt in PROMPTS:
            # pt = PyTorch；返回 shape 为 (B, T) 的 torch.Tensor
            input_ids = tokenizer.encode(
                prompt,
                return_tensors="pt",
            )

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

            print(f"\nprompt: {prompt!r}")
            print("input_ids shape:", tuple(input_ids.shape))
            print("max abs difference:", max_difference)
            print("mean abs difference:", mean_difference)
            print("our next token:", our_next_token)
            print("official next token:", official_next_token)

            assert our_next_token == official_next_token, (
                f"next token 不一致，prompt={prompt!r}"
            )

            torch.testing.assert_close(
                our_logits,
                official_logits,
                rtol=0.0,
                atol=1e-4,
            )
