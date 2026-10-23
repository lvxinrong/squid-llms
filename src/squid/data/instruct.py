import json

import torch
from torch.utils.data import Dataset

from squid.data.common import pad_to_length
from squid.tokenizer import SquidTokenizer


class InstructDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer: SquidTokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            prompt_text = format_input(entry) + "\n\n### Response:\n"
            prompt_ids = tokenizer.encode(prompt_text)
            response_ids = tokenizer.encode(entry["output"])
            self.encoded_texts.append((prompt_ids, response_ids))

    def __getitem__(self, idx) -> tuple[list[int], list[int]]:
        # 返回 (prompt_ids, response_ids)，都是 list，不转 tensor
        return self.encoded_texts[idx]

    def __len__(self) -> int:
        return len(self.encoded_texts)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

def instruct_collate(batch: list[tuple[list[int], list[int]]],
                     pad_token_id: int, ignore_index: int = -100):
    """
    输入: [(prompt_ids, response_ids), ...]
    输出: (inputs, targets)，都是 (B, L) long

    手术五步：追加 EOS → pad 到 batch 最长 → +1 切两刀
              → prompt 位打 ignore_index → 除第一个 EOS 外全 ignore
    """
    max_len = max(len(prompt) + len(response) + 1 for prompt, response in batch)
    result_inputs, result_targets = [], []

    for prompt, response in batch:
        full = torch.tensor(prompt + response + [pad_token_id], dtype=torch.long)
        padded = pad_to_length(full, max_len, pad_token_id)

        inputs = padded[:-1]
        targets = padded[1:].clone()  # 切片是视图，clone 防止污染 inputs

        # prompt 占的预测位不计分（targets 已 +1，prompt 占前 len-1 格）
        targets[:len(prompt) - 1] = ignore_index

        # 保留第一个 EOS（让模型学停止），其余 pad 位不计分
        eos_pos = (targets == pad_token_id).nonzero().squeeze(-1)
        if len(eos_pos) > 1:
            targets[eos_pos[1:]] = ignore_index

        result_inputs.append(inputs)
        result_targets.append(targets)

    return torch.stack(result_inputs), torch.stack(result_targets)






