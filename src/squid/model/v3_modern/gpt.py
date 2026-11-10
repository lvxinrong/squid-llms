from torch import nn

from squid.model.v3_modern.layers import TransformerBlock, RMSNorm
from torch.utils.checkpoint import checkpoint



MODEL_V3_DEFAULT_CONFIG = {
    "vocab_size": 151936,
    "emb_dim": 896,
    "ffn_dim": 4864,
    "n_heads": 14,
    "n_kv_heads": 2,
    "n_layers": 24,
    "rope_theta": 1000000.0,
    "eps": 1e-6,
    "qkv_bias": True,
    "drop_rate": 0.0,
    "context_length": 1024,  # 先做小规模验证
    "ffn_bias": False,
    "tie_word_embeddings": True,
}

class GPTModelV3(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = {**MODEL_V3_DEFAULT_CONFIG, **(cfg or {})}
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])],
        )

        self.final_norm = RMSNorm(cfg)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 一个 token 被理解成什么样，和它最后被生成出来时对应的向量表示，最好保持一致。
        # 直接减少参数量, 让输入和输出使用同一套语义表示
        if cfg.get("tie_word_embeddings", False):
            # 底层使用的是同一个权重张量
            # tok_emb 和 out_head 不是同一个模块，但它们共享同一个 weight 参数对象。
            self.out_head.weight = self.tok_emb.weight

        self.activation_checkpointing = cfg.get(
            "activation_checkpointing",
            False,
        )


    def forward(self, x, past_kv=None, use_cache=False):
        x = self.tok_emb(x)
        x = self.drop_emb(x)
        present = []
        for i, block in enumerate(self.trf_block):
            past = past_kv[i] if past_kv is not None else None
            if self.training and self.activation_checkpointing:
                """
                torch.utils.checkpoint 不负责保存模型文件。它是一种训练时减少激活内存占用的工具，也叫 Activation Checkpointing（激活重计算）。
                前向：
                    输入 x → 执行 block → 得到输出 y
                    但少保存一部分中间激活
                反向：
                    需要计算梯度时
                    → 用保存的输入重新执行 block
                    → 重建需要的中间激活
                    → 再计算梯度
                不一直保存所有中间结果，而是在反向传播需要时重新计算。
                普通训练会保存较多中间激活方便反向传播；torch.utils.checkpoint.checkpoint 则少保存一些，反向时再重算。
                就是时间换空间，更确切的说是 用额外的计算换显存。
                """
                x, present_kv = checkpoint(
                    block,
                    x,
                    past,
                    use_reentrant=False,
                )
            else:
                x, present_kv = block(x, past_kv=past)
            present.append(present_kv)

        x = self.final_norm(x)
        logits = self.out_head(x)
        if use_cache:
            return logits, present
        else:
            return logits
