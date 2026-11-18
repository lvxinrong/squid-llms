from pathlib import Path

from transformers import AutoTokenizer


root = Path(__file__).resolve().parents[1]
model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

messages = [
    {
        "role": "system",
        "content": "你是一个乐于助人的 AI 助手。",
    },
    {
        "role": "user",
        "content": "25 * 24 结果是多少？",
    },
    {
        "role": "assistant",
        "content": (
            "<think>\n"
            "把 24 拆成 20 + 4。\n"
            "25 × 20 = 500，25 × 4 = 100。\n"
            "500 + 100 = 600。\n"
            "</think>\n\n"
            "600"
        ),
    },
]

full_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
)

full_ids = encoded["input_ids"]
if full_ids and isinstance(full_ids[0], list):
    full_ids = full_ids[0]

pieces = tokenizer.convert_ids_to_tokens(full_ids)

print(full_text)
print(f"token count: {len(full_ids)}")

for index, (token_id, piece) in enumerate(zip(full_ids, pieces)):
    print(f"{index:>3}  {token_id:>6}  {piece!r}")
