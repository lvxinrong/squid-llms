from torch import nn

from squid.model.v3_modern.layers import TransformerBlock, RMSNorm


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


    def forward(self, x, past_kv=None, use_cache=False):
        x = self.tok_emb(x)
        x = self.drop_emb(x)
        present = []
        for i, block in enumerate(self.trf_block):
            past = past_kv[i] if past_kv is not None else None
            x, present_kv = block(x, past_kv=past)
            present.append(present_kv)

        x = self.final_norm(x)
        logits = self.out_head(x)
        if use_cache:
            return logits, present
        else:
            return logits
