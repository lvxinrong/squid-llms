import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class MemmapTokenDataset(Dataset):
    def __init__(self, token_path, max_length, stride=None, dtype="uint32",):
        self.tokens = np.memmap(
            token_path,
            dtype=dtype,
            mode="r",
        )
        self.max_length = max_length
        self.stride = max_length if stride is None else stride

        self.sample_count = len(
            range(0, len(self.tokens) - self.max_length, self.stride)
        )

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.sample_count

        if idx < 0 or idx >= self.sample_count:
            raise IndexError("样本下标越界")

        start = idx * self.stride

        input_ids = self.tokens[
            start: start + self.max_length
        ]

        target_ids = self.tokens[
            start + 1 : start + self.max_length + 1
        ]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )

def create_memmap_loader(
        token_path,
        max_length,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        stride=None,
        dtype="uint32",):
    dataset = MemmapTokenDataset(
        token_path,
        max_length=max_length,
        stride=stride,
        dtype=dtype,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )