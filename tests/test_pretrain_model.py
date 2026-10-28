import torch

from squid.data.pretrain import create_pretrain_loader
from squid.device import get_device
from squid.model.gpt import GPTModel
from squid.tokenizer import TiktokenTokenizer
from squid.train.common import plot_losses
from squid.train.pretrain import pretrain_model

GPT_CONFIG_124M = {
    "vocab_size": 50257, # BPE分词器使用的由50257个单词组成的词汇表
    "context_length": 256, # 模型通过位置嵌入能够处理的最大输入词元数量
    "emb_dim": 768, # 嵌入的维度大小，可以将每个词元转换成768维的向量
    "n_heads": 12, # 多头注意力机制中注意力头的数量
    "n_layers": 12, # 模型中 transform块数量
    "drop_rate": 0.1, # dropout的强度，0.1表示有10%的隐藏单元被随机丢弃
    "qkv_bias": False # 是否在多头注意力机制的线性层添加一个偏置向量，用于查询，键和值的计算。
}

def test_pretrain_model():
    model = GPTModel(GPT_CONFIG_124M)
    device = get_device()
    model.to(device)
    file_path = "/Users/lv/lv_repo/squid-llms/tests/test_content_file/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    tokenizer = TiktokenTokenizer("gpt2")

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)

    train_loader = create_pretrain_loader(
        train_data,
        tokenizer,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True
    )

    val_loader = create_pretrain_loader(
        val_data,
        tokenizer,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"] ,
        drop_last=False,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = pretrain_model(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,
        eval_freq=5, eval_iter=5, accum_step=5
    )
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

