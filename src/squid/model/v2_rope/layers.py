import torch
from torch import nn

from squid.model.v1_gpt.layers import LayerNorm, FeedForward
from squid.model.v2_rope.rope import (
    compute_inv_freq,
    compute_cos_sin,
    apply_rope,
)

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, rope_base=10000.0):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out 必须能被 num_heads 整除")
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1,
            ),
        )
        # inv_freq 固定不变，不需要反向传播，所以不是 nn.Parameter, 但它需要跟随模型一起 .to(device)，所以应该注册为 buffer；
        self.register_buffer(
            'inv_freq',
            compute_inv_freq(self.head_dim, base=rope_base),
            persistent=False, # persistent=False 表示不存进 checkpoint，因为加载模型时按 head_dim 和 rope_base 重新计算即可
        )
    # x.shape (B, T, C)
    def forward(self, x, past_kv=None):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        # (B, num_heads, num_token, head_dim)

        # shape (B, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_len = past_kv[0].shape[2]

        position_id = torch.arange(
            past_len, past_len + num_tokens, device=x.device
        )

        cos, sin = compute_cos_sin(position_id, self.inv_freq)
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)
        # shape (B, num_heads, num_tokens, head_dim)
        if past_kv is not None:
            past_k, past_v = past_kv
            keys = torch.cat([past_k, keys], dim =  2)
            values = torch.cat([past_v, values], dim = 2)

        present_kv = (keys, values)
        # attn_scores：  (B, H, q_len, k_len)
        attn_scores = queries @ keys.transpose(-2, -1)

        q_len = queries.shape[2]
        if q_len == 1:
            mask_bool = None
        else:
            mask_bool = self.mask.bool()[:q_len, :q_len]
        if mask_bool is not None:
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算 logits = softmax(q@key / sqrt(heads_dim))
        # (B, H, q_len, k_len)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1 # 为什么要除以 dim ** 0.5 -> Q·K 的点积会让结果的标准差膨胀 √d 倍（随机相加只按根号速度增长），除以 √d 把标准差拉回 1，防止 softmax 因输入过大而饱和。one-shot
        )
        # (B, T, num_tokens, num_tokens)
        attn_weights = self.dropout(attn_weights)
        # (B, num_tokens, T, num_tokens)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # (B, num_tokens, T, num_tokens) -> (B, num_tokens, num_head * head_dim)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        #(b, num_tokens, d_out) @ (d_out, d_out) -> (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec, present_kv


class RoPETransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.att = RoPEMultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"],
            rope_base=cfg.get("rope_base", 10000.0),
        )
        self.norm1 = LayerNorm(cfg)
        self.norm2 = LayerNorm(cfg)
        self.ffn = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, past_kv=None):

        shortcut = x
        x = self.norm1(x)
        x, present_kv = self.att(x, past_kv)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x, present_kv



