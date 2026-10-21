import torch
from mpmath import norm
from torch import nn



class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):

        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # 这里没太懂
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    # x.shape (B, T, C)
    def forward(self, x):
        # 获取x 在 q,k,v 上的投影
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # out = num_heads * head_dim
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # 计算每个头的分数，所以需要num_tokens的位置和num_heads置换一下
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim) 我认为这里就是每个头需要关注的 num_tokens和head_dim
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # 计算 attn_score  q @ key, 计算每个头的点积
        # (b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        # attn_scores.shape = (b, num_heads, num_tokens, num_tokens)
        # num_tokens, num_tokens 就是计算后的分数矩阵。 每一行就是token[i] 对于整个序列的注意力分数


        # 因果掩码，掩盖对角线右边的分数
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算 logits = softmax(q@key / sqrt(heads_dim))
        # 为什么要除以 dim ** 0.5 -> Q·K 的点积会让结果的标准差膨胀 √d 倍（随机相加只按根号速度增长），除以 √d 把标准差拉回 1，防止 softmax 因输入过大而饱和。one-shot
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        attn_weights = self.dropout(attn_weights)
        # 计算value value = logits @ values
        # 之前的张量形状是 (b, num_heads, num_tokens, num_tokens)  @ (b, num_heads, num_tokens, head_dim)
        # (attn_weights @ values) -> (b, num_heads, num_tokens, head_dim)
        # context_vec -> (b,num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 对context_vec 进行塑形
        # (b,num_tokens, num_heads, head_dim) -> (b, num_tokens, num_heads * head_dim) 但是 2和3维是被拉平了，没有计算，类似一个 (2, 2) 变换成 (1, 4)。 按行序拼成一行
        # 没有计算，只有按行序的"拼接"：原矩阵的每一行，依次变成新行（不是列）的一段。
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        # (b, num_tokens, d_out) @ (d_out, d_out) -> (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

# 前馈网络 FFN层  emb_dim -> 4 * emb_dim -> gelu -> emb_dim
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        expansion = cfg.get("ffn_expansion", 4)

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, expansion * emb_dim),
            torch.nn.GELU(),
            nn.Linear(expansion * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.eps = cfg.get("eps", 1e-5)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    # x -> (B, nums_token, emb_dim)
    # 原始特征 → 减均值除标准差（强制标准化）→ 线性变换 → 输出
    def forward(self, x):
        # 计算最后一维的平均值
        # (B, nums_token, emb_dim) -> (B, nums_token, 1)  1里面就是之前dim维的平均值
        # keepdim=True：保留被压缩的维度 也就是保留最后一个维度，不然就变成 (B, nums_token)
        mean = x.mean(dim=-1, keepdim=True)
        # x.var = ∑ (xi - mean)² / n
        # unbiased=True（默认）：样本方差，除以 N-1， unbiased=False：总体方差，除以 N
        # 计算每个token的dim的方差 每个 token 的特征方差
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # var有可能为0，所以为了防止除0，加一个很小的eps，保证分母不为0
        # 把每个 token 的特征变成 均值 0、标准差 1 的分布
        norm_x = (x - mean) / (torch.sqrt(var + self.eps))
        # 线性变换 scale和shift也是可以更新的参数
        # r = scale * norm_x + shift "缩放因子"和"偏移量"
        # 而且是每个层都会有独立的缩放因子和偏移量
        result = self.scale * norm_x + self.shift
        return result

# x -> att -> ffn -> return
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"],
        )
        self.ffn = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg)
        self.norm2 = LayerNorm(cfg)
        # nn.Dropout 受 model.train()和model.eval()控制
        # model.train() Dropout 生效，随机丢神经元
        # model.eval() Dropout 自动关闭，变成恒等映射
        # 随机丢弃的是每个token的嵌入向量中的元素，每个元素丢硬币，按照drop_rate 比例来
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    # x -> norm - > att - drop -> x + shortcut
    # x + shortcut -> norm -> ffn -> drop -> x + shortcut
    def forward(self, x):
        # 保留x，残差链接是为了保证原始x可以一直传导到最后一层。shortcut 才是每层的计算结果。
        shortcut = x
        # Pre-Norm：先 norm 再进 attention
        x = self.norm1(x)
        x = self.att(x)
        # 随机清零x中的drop_rate% 个元素，然后剩下的元素 放大 1/(1−p) 倍
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x