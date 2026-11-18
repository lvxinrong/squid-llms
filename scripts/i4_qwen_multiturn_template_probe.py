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
        "content": "我叫小吕。",
    },
    {
        "role": "assistant",
        "content": "你好，小吕。",
    },
    {
        "role": "user",
        "content": "那我叫什么？",
    },
    {
        "role": "assistant",
        "content": "你叫小吕。",
    },
]

for name, extra_kwargs in (
    ("default", {}),
    ("no_thinking", {"enable_thinking": False}),
):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **extra_kwargs,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        **extra_kwargs,
    )
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    print(f"\n===== {name} =====")
    print(text)
    print(f"token count: {len(token_ids)}")

    if name == "default":
        pieces = tokenizer.convert_ids_to_tokens(token_ids)
        for index, (token_id, piece) in enumerate(zip(token_ids, pieces)):
            print(f"{index:>3}  {token_id:>6}  {piece!r}")
