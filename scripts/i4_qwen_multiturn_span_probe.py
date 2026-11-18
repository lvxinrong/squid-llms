from pathlib import Path
from transformers import AutoTokenizer
from squid.data.chat_sft import build_assistant_targets

IM_END = "<|im_end|>"

def find_assistant_spans(tokenizer, messages, full_text, full_ids):
    spans = []
    cursor = 0

    for message in messages:
        if message["role"] != "assistant":
            continue

        content = message["content"].lstrip("\n")
        if not content:
            raise ValueError("Assistant content 不能为空")

        role_marker = f"<|im_start|>{message['role']}\n"
        role_start = full_text.index(role_marker, cursor)

        content_start = full_text.index(content, role_start + len(role_marker))
        content_end= content_start + len(content)

        end_start = full_text.index(IM_END, content_end)
        end_char = end_start + len(IM_END)

        start_ids = tokenizer.encode(
            full_text[:content_start],
            add_special_tokens = False,
        )

        end_ids = tokenizer.encode(
            full_text[:end_char],
            add_special_tokens=False,
        )

        start = len(start_ids)
        end = len(end_ids)

        assert start_ids == full_ids[:start]
        assert end_ids == full_ids[:end]

        spans.append((start, end))
        cursor = end_char

    return spans

root = Path(__file__).resolve().parents[1]
model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

messages = [
    {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
    {"role": "user", "content": "我叫小吕。"},
    {"role": "assistant", "content": "你好，小吕。"},
    {"role": "user", "content": "那我叫什么？"},
    {"role": "assistant", "content": "你叫小吕。"},
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

spans = find_assistant_spans(
    tokenizer,
    messages,
    full_text,
    full_ids,
)
targets = build_assistant_targets(full_ids, spans)
print("Assistant Full 区间：", spans)

for start, end in spans:
    print(
        f"full[{start}:{end}] "
        f"→ targets[{start - 1}:{end - 1}]"
    )

assert spans == [(29, 35), (53, 59)]
assert targets[28:34] == full_ids[29:35]
assert targets[52:58] == full_ids[53:59]

print("多轮 Assistant 区间与 targets 对齐通过")