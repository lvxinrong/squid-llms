from torch.utils.data import Dataset
from squid.data.common import pad_to_length
import torch

IM_END = "<|im_end|>"


def build_assistant_targets(full_ids, assistant_spans, ignore_index=-100):
    if not full_ids:
        raise ValueError("full_ids 不能为空")

    targets = [ignore_index] * (len(full_ids) - 1)

    for start, end in assistant_spans:
        if start < 1 or end > len(full_ids) or start >= end:
            raise ValueError(
                f"非法 Assistant 区间：[{start}, {end})"
            )

        targets[start - 1:end - 1] = full_ids[start:end]

    return targets

def get_assistant_visible_content(messages, index):
    content = messages[index]["content"].lstrip("\n")

    if index == len(messages) - 1 or "</think>" not in content:
        return content

    return content.split("</think>")[-1].lstrip("\n")


def find_assistant_spans(tokenizer, messages, full_text, full_ids):
    spans = []
    cursor = 0

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        content = get_assistant_visible_content(messages, index)
        if not content:
            raise ValueError("模板可见的 Assistant 内容不能为空")

        role_marker = f"<|im_start|>{message['role']}\n"
        role_start = full_text.index(role_marker, cursor)
        content_start = full_text.index(
            content,
            role_start + len(role_marker),
        )
        content_end = content_start + len(content)

        end_start = full_text.index(IM_END, content_end)
        end_char = end_start + len(IM_END)

        start_ids = tokenizer.encode(
            full_text[:content_start],
            add_special_tokens=False,
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


def encode_qwen_multiturn(
    tokenizer,
    messages,
    enable_thinking=None,
):
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    full_text = tokenizer.apply_chat_template(
        messages,
        **template_kwargs,
    )

    template_kwargs["tokenize"] = True
    encoded = tokenizer.apply_chat_template(
        messages,
        **template_kwargs,
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

    return full_ids[:-1], targets


class QwenMultiTurnDataset(Dataset):
    def __init__(self, data, tokenizer, enable_thinking=None):
        self.encoded_samples = []

        for entry in data:
            inputs, targets = encode_qwen_multiturn(
                tokenizer,
                entry["messages"],
                enable_thinking=enable_thinking,
            )
            self.encoded_samples.append((inputs, targets))

    def __getitem__(self, index):
        return self.encoded_samples[index]

    def __len__(self):
        return len(self.encoded_samples)


def qwen_multiturn_collate(
        batch,
        pad_token_id,
        ignore_index = -100,):
    max_len = max(len(inputs) for inputs, _ in batch)

    input_batch = []
    target_batch = []

    for inputs, targets in batch:
        input_ids = torch.tensor(inputs, dtype=torch.long)
        target_ids = torch.tensor(targets, dtype=torch.long)

        input_batch.append(pad_to_length(input_ids, max_len, pad_token_id))
        target_batch.append(pad_to_length(target_ids, max_len, ignore_index))

    return torch.stack(input_batch), torch.stack(target_batch)

