import json
from pathlib import Path

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
        responses.append({**entry, "model_response": generated[len(prompt):].replace("<|endoftext|>", "").strip()})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4, ensure_ascii=False)
    return responses

@torch.no_grad()
def generate_response_resumable(
         model,
         tokenizer,
         test_data,
         output_path,
         max_new_tokens=256,
         context_size=1024,
         temperature=0.0,
         top_k=None):
    output_path = Path(output_path)

    # 读取已有结果
    responses = []
    if output_path.exists():
        responses = json.loads(
            output_path.read_text(encoding="utf-8")
        )
        for index, saved in enumerate(responses):
            saved.setdefault("index", index)

    # 已有回答的index，不重复生成
    completed = {
        entry["index"] for entry in responses if entry.get("model_response") is not None
    }

    # 生成未完成的
    for index, entry in enumerate(tqdm(test_data, desc="Generating responses")):
        if index in completed:
            continue
        prompt = format_input(entry) + "\n\n### Response:\n"
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, context_size=context_size,
            temperature=temperature, top_k=top_k,eos_id=tokenizer.eos_id,
        )
        response = generated[len(prompt):].replace("<|endoftext|>", "").strip()
        responses.append(
            {**entry, "index": index, "model_response": response}
        )
        # 每完成一条立即落盘
        output_path.write_text(
            json.dumps(responses, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
    # 保持原有契约：即使本次没有新生成，也返回完整结果
    output_path.write_text(
        json.dumps(responses, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return responses
