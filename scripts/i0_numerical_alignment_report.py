import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from squid.checkpoints.qwen import load_qwen2_5
from squid.generate.chat import build_qwen_chat_prompt


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
WEIGHTS_PATH = (
    ROOT / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
)
ATOL = 1e-4

PROMPTS = [
    {
        "name": "chinese",
        "prompt": "你好，我叫吕炘嵘。",
    },
    {
        "name": "english",
        "prompt": "The capital of France is",
    },
    {
        "name": "code",
        "prompt": (
            "def find_even_numbers(numbers):\n"
            "    even_numbers = []\n"
        ),
    },
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_metrics(our_tensor, official_tensor):
    assert our_tensor.shape == official_tensor.shape
    difference = (our_tensor - official_tensor).abs()

    return {
        "shape": list(our_tensor.shape),
        "our_finite": bool(torch.isfinite(our_tensor).all()),
        "official_finite": bool(
            torch.isfinite(official_tensor).all()
        ),
        "max_abs_difference": difference.max().item(),
        "mean_abs_difference": difference.mean().item(),
        "allclose": torch.allclose(
            our_tensor,
            official_tensor,
            rtol=0.0,
            atol=ATOL,
        ),
    }


def logits_metrics(our_logits, official_logits):
    result = tensor_metrics(our_logits, official_logits)
    our_next = our_logits[0, -1].argmax().item()
    official_next = official_logits[0, -1].argmax().item()

    result.update({
        "our_next_token": our_next,
        "official_next_token": official_next,
        "next_token_match": our_next == official_next,
    })
    result["passed"] = (
        result["our_finite"]
        and result["official_finite"]
        and result["allclose"]
        and result["next_token_match"]
    )
    return result


def build_chat_prompt(tokenizer):
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
    return build_qwen_chat_prompt(
        tokenizer,
        history,
        current_input,
    )


def main():
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    our_model = load_qwen2_5(
        WEIGHTS_PATH,
        device=device,
    )
    our_model.eval()

    official_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    official_model.eval()

    cases = [
        *PROMPTS,
        {
            "name": "multiturn_chat",
            "prompt": build_chat_prompt(tokenizer),
        },
    ]
    full_results = []

    with torch.no_grad():
        for case in cases:
            input_ids = tokenizer.encode(
                case["prompt"],
                return_tensors="pt",
            )
            our_logits = our_model(input_ids)
            official_logits = official_model(input_ids).logits

            result = logits_metrics(
                our_logits,
                official_logits,
            )
            result.update({
                "name": case["name"],
                "prompt": case["prompt"],
                "input_ids": input_ids[0].tolist(),
            })
            full_results.append(result)

        cache_input = tokenizer.encode(
            PROMPTS[0]["prompt"],
            return_tensors="pt",
        )
        our_logits, our_cache = our_model(
            cache_input,
            use_cache=True,
        )
        official_output = official_model(
            cache_input,
            use_cache=True,
        )
        official_logits = official_output.logits
        official_cache = official_output.past_key_values

        prefill = logits_metrics(
            our_logits,
            official_logits,
        )
        kv_layers = []

        assert len(our_cache) == len(official_cache.layers)
        for layer_index in (0, len(our_cache) - 1):
            our_key, our_value = our_cache[layer_index]
            official_layer = official_cache.layers[layer_index]

            kv_layers.append({
                "layer_index": layer_index,
                "key": tensor_metrics(
                    our_key,
                    official_layer.keys,
                ),
                "value": tensor_metrics(
                    our_value,
                    official_layer.values,
                ),
            })

        decode_steps = []

        for step in range(3):
            next_token = official_logits[:, -1].argmax(
                dim=-1,
                keepdim=True,
            )
            input_token_id = next_token.item()

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

            result = logits_metrics(
                our_logits,
                official_logits,
            )
            result.update({
                "step": step + 1,
                "input_token_id": input_token_id,
                "our_cache_length": our_cache[0][0].shape[2],
                "official_cache_length": (
                    official_cache.layers[0].keys.shape[2]
                ),
            })
            decode_steps.append(result)

    kv_passed = all(
        metrics["our_finite"]
        and metrics["official_finite"]
        and metrics["allclose"]
        for layer in kv_layers
        for metrics in (layer["key"], layer["value"])
    )
    cache_decode_passed = all(
        step["passed"]
        and step["our_cache_length"] == step["official_cache_length"]
        for step in decode_steps
    )
    full_passed = all(case["passed"] for case in full_results)

    summary = {
        "full_logits_cases": len(full_results),
        "full_logits_passed": sum(
            case["passed"]
            for case in full_results
        ),
        "cache_prefill_passed": prefill["passed"],
        "cache_layers_checked": len(kv_layers),
        "cache_kv_passed": kv_passed,
        "cache_decode_steps": len(decode_steps),
        "cache_decode_passed": sum(
            step["passed"]
            and step["our_cache_length"] == step["official_cache_length"]
            for step in decode_steps
        ),
        "status": (
            "pass"
            if full_passed
            and prefill["passed"]
            and kv_passed
            and cache_decode_passed
            else "fail"
        ),
    }

    report = {
        "schema_version": "i0-numerical-alignment-v1",
        "config": {
            "model": MODEL_NAME,
            "weights_path": str(WEIGHTS_PATH.relative_to(ROOT)),
            "weights_sha256": file_sha256(WEIGHTS_PATH),
            "device": "cpu",
            "dtype": "torch.float32",
            "official_attention": (
                official_model.config._attn_implementation
            ),
            "local_files_only": True,
            "rtol": 0.0,
            "atol": ATOL,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "summary": summary,
        "full_logits": full_results,
        "cache": {
            "prefill": prefill,
            "kv_layers": kv_layers,
            "decode_steps": decode_steps,
        },
        "boundary": (
            "只覆盖固定输入、CPU、FP32、官方 eager 和当前权重；"
            "不推广到任意输入、设备、dtype、SDPA 或生成质量。"
        ),
    }

    output_path = ROOT / "data/reports/i0-numerical-alignment.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {output_path}")

    assert summary["status"] == "pass"


if __name__ == "__main__":
    main()
