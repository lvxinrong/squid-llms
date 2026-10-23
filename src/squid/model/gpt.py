import torch
from torch import nn
from squid.model.layers import TransformerBlock, LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ## nn.Embedding 创建了一个可学习的浮点矩阵。每个元素是随机数，默认从标准正态分布 N(0, 1)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg)

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    # (B, T)  (batch, token)
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.size()
        # (vocab_size, emb_dim)
        # nn.Embedding(x) 查表 查询每个token在tok_emb 中的词嵌入向量
        # 所以这里的tok_emb.shape(B, T, emb_dim) 相当于在原始词表中查询每个token_id 初始化后的token_embedding
        tok_embeds = self.tok_emb(in_idx)
        # pos_embeds.shape(seq_len, emb_dim)
        # 给每个位置取一个位置向量，位置和token无关，pos是座位，token只是某个批次坐在这个位置上的人，批次更换，位置上的token也会变。所以
        # pos_emb的声明是 nn.Embedding(context_length, emb_dim)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        # 把位置信息编织进tok_embeds。x就同时拥有token和位置的信息
        x = tok_embeds + pos_embeds
        # 提前接触不完整数据，逼它学鲁棒的语义表示，而不是死记 token 的完整 embedding
        # x.shape (B, T, emb_dim)
        x = self.drop_emb(x)
        # 经过多个trf block
        # x = x + att(norm1(x))    # 残差相加时，分支输出被 norm 过，但 x 本身没被 norm
        # x = x + ffn(norm2(x))
        x = self.trf_block(x)
        # 最终的结果归一化
        # 因为trf block里面 att ffn的norm都是针对shortcut 而不是主干x。这里是针对快捷连接的主干x进行最后一次归一化。
        # 残差主干 x 从头到尾没有被归一化过。它一路只做"原始值 + 分支增量"的叠加，层数一深，数值尺度会越滚越大（方差随深度线性增长）。
        # Pre-Norm 把归一化推迟了 N 层，final_norm 就是最后的"总闸"——不拉这一下，softmax 直接饱和，模型白训。
        x = self.final_norm(x)
        # 转换成
        # out_head.shape (emb_dim, vocab_size)
        # (B, T, emb_dim) @ (emb_dim, vocab_size) ->  (B, T, vocab_size)
        logits = self.out_head(x)
        # logits 记录的是每个T的next-token在整个vocab中的分数，假设vocab_size = 50257, 那最后一维就是记录的某个token的next-token 在词表中的分数分布，50257个词每个词有个分数，这里是分数，还没有到计算概率的时候
        return logits