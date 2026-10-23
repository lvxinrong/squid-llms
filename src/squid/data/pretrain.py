import torch
from torch.utils.data import Dataset, DataLoader

from squid.tokenizer import SquidTokenizer


class PretrainDataset(Dataset):
    def __init__(self, text: str, tokenizer: SquidTokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        text_ids = tokenizer.encode(text)
        # stride 是重叠长度，前一个序列和后一个序列重叠多少，一般是max_length的一半
        # 重叠不只"解决上下文割裂"，它还让同样一篇文本切出更多样本（stride 越小样本越多），数据利用率上去了——但代价是训练步数变多、而且样本之间近似重复，边际收益递减。所以 stride 是个旋钮
        # 不是越大越好也不是越小越好，一半是个经验上的甜点。
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
