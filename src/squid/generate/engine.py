import torch
from tqdm import tqdm

from squid.data.instruct import format_input


# idx.shape (B, T)
#
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k = None, eos_id = None):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
        logits, past_kv = model(idx_cond, use_cache=True)
    # 最大可以生产的token，每次循环生成一个next
    for _ in range(max_new_tokens):
        # 获取最新个context_size长度的token
        idx_next = sample_next(logits[:, -1, :], temperature=temperature, top_k=top_k)
        idx = torch.cat([idx, idx_next], dim=1)
        if idx_next == eos_id:
            break
        with torch.no_grad():
            logits, past_kv = model(idx_next, past_kv=past_kv,use_cache=True)
    return idx

def sample_next(logits, temperature=0.0, top_k=None):
    if top_k is not None:
        logits_topk, _ = torch.topk(logits, k=top_k)
        # 倒排，就是获取topk位置的分数，后续需要遮盖掉 < min
        min_val = logits_topk[:, -1]
        # 取topk个token的分数
        logits = torch.where(
            logits < min_val,
            torch.tensor(float("-inf")).to(logits.device),
            logits
        )
    # 温度控制， 0<t<1 缩放比例尖锐，分数高的会更加突出。 t > 1 缩放比例温和，分数低的词分数会提高
    # 假设两个词的分数是 10, 20
    # 温度是 0.5   分数就变成20， 40  差值扩大了一倍
    # 温度是 2     分数就变成 5， 10  差值减少了一倍
    if temperature > 0.0:
        logits = logits / temperature
        probas = torch.softmax(logits, dim=-1)
        # 概率采样， 给logits中分数小的token留个可能得机会
        idx_next = torch.multinomial(probas, num_samples=1)
    else:
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
    return idx_next

def generate_no_cache(model, idx, max_new_tokens, context_size, temperature=0.0, top_k = None, eos_id = None):
    # 最大可以生产的token，每次循环生成一个next
    for _ in range(max_new_tokens):
        # 获取最新个context_size长度的token
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits =  model(idx_cond, use_cache=False)
        # 取最后一个token的logits 向量，用于获取next-token
        logits = logits[:, -1, :]
        idx_next = sample_next(logits, temperature=temperature, top_k=top_k)
        if idx_next == eos_id:
            break
        # 很关键，把预测的下一个词也要加入到idx中，用于预测next-next的token。
        idx = torch.cat([idx, idx_next], dim=1)
    return idx