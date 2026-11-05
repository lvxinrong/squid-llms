import torch
from torch.utils.data import Dataset, DataLoader

from squid.tokenizer import SquidTokenizer


class PretrainDataset(Dataset):
    def __init__(self, text: str, tokenizer: SquidTokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        text_ids = tokenizer.encode(text)
        # stride 是相邻窗口的起始位置间隔；重叠长度为 max_length - stride。
        # stride 越小，样本越多且近似重复越多；stride = max_length 时窗口不重叠。
        for i in range(0, len(text_ids) - max_length, stride):
            self.input_ids.append(text_ids[i: i + max_length])
            self.target_ids.append(text_ids[i + 1 : i + max_length + 1])


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx]),
            torch.tensor(self.target_ids[idx])
        )

# shuffle 从 token_ids 中取序列的方式，false按照seq严格方式来获取，一般用于test验证
# true 按照随机的方式取
# 打乱之后每个 batch 都是随机抽样的混合，梯度才是无偏的。而验证集要的是可复现，顺序无所谓但必须每次一样，所以 False。
# drop_last=True 让 DataLoader 丢掉最后一个装不满的 batch，而不是把它当作一个小 batch 交出去。
def create_pretrain_loader(text, tokenizer, max_length, stride,
                                batch_size=4, shuffle=True, drop_last=True) -> DataLoader:
    dataset = PretrainDataset(text, tokenizer, max_length=max_length, stride=stride)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last)
