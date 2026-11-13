import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from squid.model.v4_qwen3.layers import (
    RMSNorm,
    TransformerBlock,
)

MODEL_V4_DEFAULT_CONFIG = {
    "vocab_size": 151936,
    "emb_dim": 1024,
    "ffn_dim": 3072,
    "n_heads": 16,
    "n_kv_heads": 8,
    "n_layers": 28,
    "head_dim": 128,
    "rope_theta": 1000000.0,
    "eps": 1e-6,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "drop_rate": 0.0,
    "context_length": 1024,
    "ffn_bias": False,
    "tie_word_embeddings": True,
}

class GPTModelV4(nn.Module):
    supports_position_offset = True
    def __init__(self, cfg = None):
        super().__init__()
        cfg = {
            **MODEL_V4_DEFAULT_CONFIG,
            **(cfg or {}),
        }

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])],
        )

        self.final_norm = RMSNorm(cfg)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if cfg.get("tie_word_embeddings", False):
            self.out_head.weight = self.tok_emb.weight

        self.activation_checkpointing = cfg.get(
            "activation_checkpointing",
            False,
        )

    def forward(self, x, past_kv = None, use_cache = False, position_offset = None):
        x = self.tok_emb(x)
        x = self.drop_emb(x)

        present = []

        for index, block in enumerate(self.trf_block):
            past = past_kv[index] if past_kv is not None else None
            if self.training and self.activation_checkpointing:
                x, present_kv = checkpoint(block, x, past, position_offset, use_reentrant = False)
            else:
                x, present_kv = block(x, past_kv = past, position_offset = position_offset)

            present.append(present_kv)

        x = self.final_norm(x)
        logits = self.out_head(x)

        if use_cache:
            return logits, present
        return logits
