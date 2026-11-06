from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from squid.checkpoints.qwen import load_qwen2_5

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def assert_logits_close(our_logits, official_logits, label):
    assert our_logits.shape == official_logits.shape
    # TODO torch.isfinite的含义
    assert torch.isfinite(our_logits).all()
    assert torch.isfinite(official_logits).all()
    # abs 对张量逐元素求绝对值，返回一个形状相同的新张量。
    # [-1.0, 2.0, -3.0].abs() -> [1.0, 2.0, 3.0]
    difference = (our_logits - official_logits).abs()
    our_next_token = our_logits[0, -1].argmax().item()
    official_next_token = official_logits[0, -1].argmax().item()

    print(f"\n{label}")
    print("max abs difference:", difference.max().item())
    print("mean abs difference:", difference.mean().item())
    print("our next token:", our_next_token)
    print("official next token:", official_next_token)

    assert our_next_token == official_next_token

    torch.testing.assert_close(
        our_logits,
        official_logits,
        rtol=0.0,
        atol=1e-4,
    )


def test_qwen_logits_compare_with_official():
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    our_model = load_qwen2_5(
        ROOT / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors",
        device=device,
    )

    official_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype = torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    official_model.eval()

    prompt = "你好，我叫吕炘嵘"

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        our_logits, our_cache = our_model(input_ids, use_cache=True,)
        official_output = official_model(
            input_ids,
            use_cache=True,
        )
        official_logits = official_output.logits
        official_cache = official_output.past_key_values

        assert_logits_close(our_logits, official_logits, "prefill")

        assert len(our_cache) == len(official_cache.layers) == 24

        for layer_index in (0, 23):
            our_key, our_value = our_cache[layer_index]
            official_layer = official_cache.layers[layer_index]

            assert our_key.shape == official_layer.keys.shape
            assert our_value.shape == official_layer.values.shape
            # rtol 和 atol的含义是啥
            torch.testing.assert_close(
                our_key,
                official_layer.keys,
                rtol=0.0,
                atol=1e-4,
            )

            torch.testing.assert_close(
                our_value,
                official_layer.values,
                rtol=0.0,
                atol=1e-4,
            )

        for step in range(3):
            next_token = official_logits[:, -1].argmax(
                dim=-1,
                keepdim=True,
            )

            our_logits, our_cache = our_model(
                next_token,
                past_kv=our_cache,
                use_cache=True,
            )

            official_output = official_model(
                next_token,
                past_key_values=official_cache,
                use_cache=True,
            )

            official_logits = official_output.logits
            official_cache = official_output.past_key_values

            assert_logits_close(
                our_logits,
                official_logits,
                f"cache decode step {step + 1}",
            )

            print("cache length:", our_cache[0][0].shape[2])