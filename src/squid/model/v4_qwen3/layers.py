import torch
from torch import nn
from torch.nn import functional as F
from squid.model.v4_qwen3.rope import (
    apply_rope,
    compute_cos_sin,
    compute_inv_freq,
)

"""
    x 进入
    → x.float()
    → 平方、均值、rsqrt、scale 都使用 FP32
    → 归一化结果转回 input_dtype
    
    这样可以兼顾：
        中间计算的数值范围与稳定性；
        外部混合精度链路的类型一致性。
"""
class RMSNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eps = cfg.get("eps", 1e-6)
        self.scale = nn.Parameter(
            torch.ones(cfg["emb_dim"])
        )

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()

        mean_square = x_float.pow(2).mean(
            dim = -1,
            keepdim = True,
        )

        y = x_float * torch.rsqrt(mean_square + self.eps) * self.scale.float()

        return y.to(dtype=input_dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.gate_proj = nn.Linear(cfg["emb_dim"], cfg["ffn_dim"], bias = cfg.get("ffn_bias", False))
        self.up_proj = nn.Linear(cfg["emb_dim"], cfg["ffn_dim"], bias = cfg.get("ffn_bias", False))
        self.down_proj = nn.Linear(cfg["ffn_dim"], cfg["emb_dim"], bias = cfg.get("ffn_bias", False))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(
            self.act(self.gate_proj(x)) * self.up_proj(x)
        )

"""
    Q/K/V 投影
    → Q/K 按 head 归一化
    → 转成 (B, H, T, D)
    → RoPE
    → 拼接历史 KV Cache
    → GQA 扩展
    → 因果掩码
    → attention
    → out_proj 降回 emb_dim
"""
class Qwen3Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.n_heads = int(cfg["n_heads"])
        self.n_kv_heads = int(cfg["n_kv_heads"])
        self.emb_dim = int(cfg["emb_dim"])
        self.head_dim = int(cfg["head_dim"])

        assert self.n_heads % self.n_kv_heads == 0

        self.num_queries_per_kv = self.n_heads // self.n_kv_heads

        attention_bias = cfg.get("attention_bias", False)

        self.W_query = nn.Linear(self.emb_dim, self.n_heads * self.head_dim, bias = attention_bias)
        self.W_key = nn.Linear(self.emb_dim, self.n_kv_heads * self.head_dim, bias = attention_bias)
        self.W_value = nn.Linear(self.emb_dim, self.n_kv_heads * self.head_dim, bias = attention_bias)

        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.emb_dim, bias = attention_bias)
        # QK RMSNorm 是给每个 head 的 Q/K 控制数值尺度，让 attention 匹配更稳定。更细粒度的归一化
        self.q_norm = RMSNorm({
            "emb_dim": self.head_dim,
            "eps": cfg.get("eps", 1e-6),
        })

        self.k_norm = RMSNorm({
            "emb_dim": self.head_dim,
            "eps": cfg.get("eps", 1e-6),
        })

        self.dropout = nn.Dropout(cfg.get("attention_dropout", cfg.get("drop_rate", 0.0),))

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    cfg["context_length"],
                    cfg["context_length"],
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq",
            compute_inv_freq(
                self.head_dim,
                base=cfg.get("rope_theta", 10000.0),
            ),
            persistent=False,
        )

    def expand_kv(self, x):
        if self.num_queries_per_kv == 1:
            return x

        b, n_kv_heads, num_tokens, head_dim = x.shape
        x = x[:, :, None, :, :].expand(
            b,
            n_kv_heads,
            self.num_queries_per_kv,
            num_tokens,
            head_dim,
        )
        return x.reshape(
            b,
            n_kv_heads * self.num_queries_per_kv,
            num_tokens,
            head_dim,
        )

    def forward(self, x, past_kv = None, position_offset = None):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x).view(b, num_tokens, self.n_heads, self.head_dim)
        keys = self.W_key(x).view(b, num_tokens, self.n_kv_heads, self.head_dim)
        values = self.W_value(x).view(b, num_tokens, self.n_kv_heads, self.head_dim)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_len = past_kv[0].shape[2]

        if position_offset is None:
            position_offset = past_len

        position_ids = torch.arange(
            position_offset,
            position_offset + num_tokens,
            device = x.device,
        )

        cos ,sin = compute_cos_sin(position_ids, self.inv_freq)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            # past_k:    (B, KV heads, past_tokens, head_dim)
            # 当前 keys: (B, KV heads, new_tokens,  head_dim)
            keys = torch.cat([past_k, keys], dim = 2)
            values = torch.cat([past_v, values], dim = 2)

        present_kv = (keys, values)

        keys = self.expand_kv(keys)
        values = self.expand_kv(values)

        q_len = queries.shape[2]
        k_len = keys.shape[2]

        if q_len == 1:
            mask_bool = None
        else:
            past_len = k_len - q_len
            mask_bool = self.mask[past_len:past_len + q_len, :k_len]

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / self.head_dim ** 0.5

        if mask_bool is not None:
            attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores.float(), dim = -1).to(dtype=queries.dtype)

        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.n_heads * self.head_dim)

        return self.out_proj(context_vec), present_kv


"""
    x
    → norm1 → attention → 残差相加
    → norm2 → SwiGLU FFN → 残差相加
    → 输出与当前层 present_kv
"""
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = Qwen3Attention(cfg)
        self.ffn = SwiGLUFFN(cfg)
        self.norm1 = RMSNorm(cfg)
        self.norm2 = RMSNorm(cfg)

        self.drop_shortcut = nn.Dropout(cfg.get("drop_rate", 0.0))


    def forward(self, x, past_kv = None, position_offset = None):
        shortcut = x

        x = self.norm1(x)
        x, present_kv = self.att(x, past_kv = past_kv, position_offset = position_offset)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x, present_kv






