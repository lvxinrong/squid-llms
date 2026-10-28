import torch
from torch.utils.data import DataLoader, Dataset

def pack_tokens(texts: list[str], tokenizer, eos_id: int) -> list[int]:
    """把多篇文本编码后拼接成一条连续 token 流。
    每篇 encode 之后，在末尾追加一个 eos_id 作为文档边界，
    然后按顺序首尾相接。返回一维的 id 列表。
    """
    ids = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        token_ids.append(eos_id)
        ids.extend(token_ids)
    return ids

def create_stream_loader(ids: list[int], context_length: int, batch_size: int,
                         shuffle: bool = True,drop_last: bool=True, stride: int = None) -> DataLoader:
    """把连续 token 流切成定长样本。
     复用 PretrainDataset 的思路：input 和 target 错位一位，
     用默认 collate 组 batch。
     """
    dataset = TokenStreamDataset(ids, context_length, context_length if stride is None else stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


class TokenStreamDataset(Dataset):
    def __init__(self, ids: list[int], max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(ids) - max_length, stride):
            self.input_ids.append(ids[i: i + max_length])
            self.target_ids.append(ids[i + 1 : i + max_length + 1])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx]),
            torch.tensor(self.target_ids[idx])
        )
