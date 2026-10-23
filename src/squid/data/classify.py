import torch
from pandas import DataFrame
from torch.utils.data import Dataset
import pandas as pd

from squid.data.common import pad_to_length
from squid.tokenizer import SquidTokenizer

# 加载已经有了text和label的数据
# names=["Label", "Text"]
class ClassifyDataset(Dataset):
    def __init__(self, df: DataFrame, tokenizer: SquidTokenizer, max_length: int,
                 label2id: dict | None = None, balance: bool = False):
        if label2id is None:
            label2id = {label: i for i, label in enumerate(sorted(df["Label"].unique()))}
        self.label2id = label2id

        if balance:
            min_count = df["Label"].value_counts().min()
            df = (df.groupby("Label", group_keys=False)
                  .apply(lambda g: g.sample(min_count, random_state=123)))

        # 3. 编码 + 截断，存成两个 list
        self.texts = [tokenizer.encode(t)[:max_length] for t in df["Text"]]
        self.labels = [label2id[label] for label in df["Label"]]
        assert len(self.texts) == len(self.labels), "texts 和 labels 数量不一致"

    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

    def __len__(self):
        return len(self.labels)

"""
 输入: [(text_tensor, label), ...]  # 长度不一
 输出: (inputs (B, max_len) long, labels (B,) long)
 步骤: 找 max_len → pad_to_length 逐条补 → stack → labels stack
"""
def classify_collate(batch, pad_id=50256):
    texts = [text for text, _ in batch]
    labels = [label for _, label in batch]

    max_len = max(len(t) for t in texts)
    inputs = torch.stack([pad_to_length(t, max_len, pad_id) for t in texts])
    return inputs, torch.stack(labels)


