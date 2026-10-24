import json

import torch
from tqdm import tqdm

from squid.data.instruct import format_input
from squid.generate.text import generate_text


@torch.no_grad()
def generate_responses(model, tokenizer, test_data, output_path,
                       max_new_tokens=256, context_size=1024,
                       temperature=0.0, top_k=None):
    """批量生成测试集回答，返回带 model_response 字段的新列表并落盘。

    prompt 与训练同模板（含 ### Response: 头），生成后切掉 prompt 只留回答。
    """
    responses = []
    for entry in tqdm(test_data, desc="Generating responses"):
        prompt = format_input(entry) + "\n\n### Response:\n"
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, context_size=context_size,
            temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id,
        )
        responses.append({**entry, "model_response": generated[len(prompt):].strip()})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4, ensure_ascii=False)
    return responses
