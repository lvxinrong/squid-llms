from torch import nn

from squid.model.v1_gpt.layers import LayerNorm
from squid.model.v2_rope.layers import RoPETransformerBlock

class GPTModelRoPE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # RoPE 在每层 MHA 内处理位置，不再创建 pos_emb
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_block = nn.Sequential(
            *[RoPETransformerBlock(cfg) for _ in range(cfg["n_layers"])],
        )

        self.final_norm = LayerNorm(cfg)

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

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