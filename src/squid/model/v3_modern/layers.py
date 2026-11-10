import torch
from torch import nn
from squid.model.v2_rope.rope import (
    compute_inv_freq,
    compute_cos_sin,
    apply_rope,
)
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb_dim = cfg["emb_dim"]
        self.eps = cfg.get("eps", 1e-6)
        self.scale = nn.Parameter(torch.ones(self.emb_dim))

    """
    LayerNorm 是“减均值、除标准差”，RMSNorm 是“不减均值，直接除以均方根”。
    LayerNorm 先把数据中心化，再控制方差；RMSNorm 不管中心点，只控制向量的整体幅度。
    两者都在控制激活值的整体幅度，防止经过多次计算后某些特征向量过大或过小，从而让训练更稳定。
    RMSNorm 主要通过等比例缩放控制幅度，LayerNorm 还会先做均值中心化。
    """
    def forward(self, x):
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        y = x * torch.rsqrt(mean_square + self.eps) * self.scale
        return y

class SwiGLUFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ffn_bias = cfg.get("ffn_bias", False)
        self.emb_dim = int(cfg["emb_dim"])
        self.hidden_dim = int(cfg.get("ffn_dim", 3 * cfg["emb_dim"]))
        self.gate_proj = nn.Linear(self.emb_dim, self.hidden_dim, bias=ffn_bias)
        self.up_proj = nn.Linear(self.emb_dim, self.hidden_dim, bias=ffn_bias)
        self.down_proj = nn.Linear(self.hidden_dim, self.emb_dim, bias=ffn_bias)
        self.act = nn.SiLU()

    """
    y = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    gate_proj门控信号，SiLU 之后才形成实际的连续开度。
    up_proj 升维后的x (B, T, emb_dim) -> (B, T, hidden_dim)  hidden_dim 默认是emb_dim的3倍
    down_proj 降维回 (B, T, hidden_dim) -> (B, T, emb_dim)

    SwiGLUFFN 相比较之前的 FeedForward 核心是多了一个门阀矩阵。
    按照我的理解，这个门阀矩阵是类似一个电阻的装置，允许电流通过大小，也就是hidden_dim每个维度允许通过多少

    但是核心原理和FeedForward不变，依然是在token经过attention 输出后的每个 token 的隐藏表示 做逐位置变换
    再更高维度对自身的隐藏表示进行编辑。
    阀门也就是在维度升高之后，增加了一个可以由模型自行通过学习修改阀门流量的开关。可以决定这些信息哪些值得保留，放大或者压下去。
    SwiGLU FFN 在升维后，是用 gate_proj 单独生成一路门控信号，经 SiLU 变成每个 hidden 维度的连续开度，再与 up_proj 的内容逐元素相乘。这个开度可学习、可大于 1、可接近 0、也可为负，因此模型能逐维度决定信息保留、放大、压制甚至反向。
    """
    def forward(self, x):
        return self.down_proj(
            self.act(self.gate_proj(x)) * self.up_proj(x)
        )

"""
核心是减少了KV的头，Q的头保持不变

假设Q = 16, K,V = 4
所以按照我的理解，GQA是针对推理时候的优化，而不是训练的时候
KV的数量后，为了可以和Q进行矩阵运算，是进行的横向复制。那么就可以推出计算量和数据量并没有减少 (1份复制到4份)
但是k，v的数据量下降到原本的四分之一了
也就是在推理的时候，我们接入了kv cache的空间占用量下降了75%
所以kv cache中保留的未扩展的4个头。每个Token的KV存储量大约是原来的1/4

GQA 主要通过减少 K/V 头数来降低推理时的 cache 和带宽压力，而不是直接减少所有训练或推理计算。
"""
class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = int(cfg["n_heads"])
        self.n_kv_heads = int(cfg["n_kv_heads"])
        self.emb_dim = int(cfg["emb_dim"])
        assert self.emb_dim % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        self.head_dim = self.emb_dim // self.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        # 每组 K/V 对应多少个 Q 头
        self.num_queries_per_kv = self.n_heads // self.n_kv_heads
        self.W_query = nn.Linear(self.emb_dim, self.emb_dim, bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(self.emb_dim, self.kv_dim, bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(self.emb_dim, self.kv_dim, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.dropout = nn.Dropout(cfg["drop_rate"])
        # 创建一个因果掩码（causal mask）张量，并把它注册成 Module 的 buffer
        # register_buffer 是给“不是可学习参数、但属于模型状态、需要跟着模型一起搬设备/转精度/存 state_dict”的张量用的。
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(cfg["context_length"], cfg["context_length"], dtype=torch.bool),
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
        # 用于切换eager和SDPA的开关
        self.attention_impl = cfg.get("attention_impl", "eager")

    def expand_kv(self, x):
        if self.num_queries_per_kv == 1:
            return x
        b, n_kv_heads, num_tokens, head_dim = x.shape
        x = x[:, :, None, :, :].expand(
            b, n_kv_heads, self.num_queries_per_kv, num_tokens, head_dim
        )
        return x.reshape(b, n_kv_heads * self.num_queries_per_kv, num_tokens, head_dim)

    """
    (B, T, K/V, D) 横向扩展复制成 (B, T, Q, D)
    假设Q = 16， k,v = 4
    也就是针对K进行横向复制扩展，每份复制4份，这样后面才能进行 Q @ K.T的运算，不然 Q和KV的维度是对不上的
    K0, K1, K2, K3 转换成
    (
    (K0, K0, K0, K0),
    (K1, K1, K1, K1),
    ...
    )
    """
    def forward(self, x, past_kv=None):
        b, num_tokens, _ = x.shape
        # (B, T, C) -> (B, T, H, D)
        queries = self.W_query(x).view(b, num_tokens, self.n_heads, self.head_dim)
        # (B, T, C) -> (B, T, KV, D)
        keys = self.W_key(x).view(b, num_tokens, self.n_kv_heads, self.head_dim)
        # (B, T, C) -> (B, T, KV,D)
        values = self.W_value(x).view(b, num_tokens, self.n_kv_heads, self.head_dim)

        #(B, T, H, D) -> (B, H, T, D)
        queries = queries.transpose(1, 2)
        #(B, T, H, D) -> (B, KV, T, D)
        keys = keys.transpose(1, 2)
        # (B, T, H, D) -> (B, KV, T, D)
        values = values.transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_len = past_kv[0].shape[2]

        position_ids = torch.arange(
            past_len,
            past_len + num_tokens,
            device=x.device,
        )

        cos, sin = compute_cos_sin(position_ids, self.inv_freq)
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            keys = torch.cat([past_k, keys], dim=2)
            values = torch.cat([past_v, values], dim=2)

        present_kv = (keys, values)
        keys = self.expand_kv(keys)
        values = self.expand_kv(values)

        q_len = queries.shape[2]
        k_len = keys.shape[2]
        # 确保每个位置的计算都符合因果顺序，并让 prefill 产生的 cache 与后续逐 token decode 保持一致。
        if q_len == 1:
            mask_bool = None
        else:
            past_len = k_len - q_len
            mask_bool = self.mask[
                past_len:past_len + q_len,
                :k_len,
            ]

        if self.attention_impl == "sdpa":
            # 我们的 mask 是“True 就屏蔽”，SDPA 的 bool mask 是“True 就允许”，所以传递前要用 ~ 做逐元素真假取反。
            attn_mask = None if mask_bool is None else ~mask_bool

            context_vec = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        elif self.attention_impl == "eager":
            attn_scores = queries @ keys.transpose(-2, -1)

            if mask_bool is not None:
                attn_scores = attn_scores.masked_fill(
                    mask_bool,
                    -torch.inf,
                )

            attn_weights = torch.softmax(
                attn_scores / self.head_dim ** 0.5,
                dim=-1,
            )
            attn_weights = self.dropout(attn_weights)
            context_vec = attn_weights @ values
        else:
            raise ValueError(f"未知 attention_impl: {self.attention_impl}")

        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.n_heads * self.head_dim
        )
        return self.out_proj(context_vec), present_kv

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(cfg)
        self.ffn = SwiGLUFFN(cfg)
        self.norm1 = RMSNorm(cfg)
        self.norm2 = RMSNorm(cfg)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, past_kv=None):
        shortcut = x

        x = self.norm1(x)
        x, present_kv = self.att(x, past_kv)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x, present_kv