import time

import torch
from transformers import AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device, memory_snapshot
from squid.generate.chat import build_qwen_chat_prompt
from squid.generate.engine import generate


model_dir = "data/model_pth/qwen3/qwen3-4b"
device = get_device()

print("device:", device, flush=True)
print("loading model...", flush=True)
load_start = time.perf_counter()

model = load_qwen3_dir(
    model_dir,
    context_size=4096,
    device=device,
    dtype=torch.bfloat16,
)

print(
    "model ready; seconds:",
    round(time.perf_counter() - load_start, 2),
    flush=True,
)
print(
    "parameters:",
    sum(parameter.numel() for parameter in model.parameters()),
    flush=True,
)
print(
    "dtype:",
    next(model.parameters()).dtype,
    flush=True,
)
print("memory:", memory_snapshot(device), flush=True)

print("loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = build_qwen_chat_prompt(
    tokenizer,
    [],
    "只输出 OK",
    enable_thinking=False,
)
input_ids = tokenizer.encode(
    prompt,
    return_tensors="pt",
).to(device)

print("generating 8 tokens...", flush=True)
generate_start = time.perf_counter()

output = generate(
    model,
    input_ids,
    max_new_tokens=8,
    context_size=4096,
    temperature=0.0,
    eos_id=tokenizer.eos_token_id,
)

print(
    "generation seconds:",
    round(time.perf_counter() - generate_start, 2),
    flush=True,
)

generated = output[0, input_ids.shape[1]:]
print(
    "response:",
    tokenizer.decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ),
    flush=True,
)
print("memory:", memory_snapshot(device), flush=True)
