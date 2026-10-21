import torch

# idx.shape (B, T)
# 
def generate(model, idx, max_new_tokens, context_size, temperature, top_k = None, eos_id = None):
    # 最大可以生产的token，每次循环生成一个next
    for _ in range(max_new_tokens):
        # 获取最新个context_size长度的token
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        # 取最后一个token的logits 向量，用于获取next-token
        logits = logits[:, -1, :]

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

        if idx_next == eos_id:
            break
        # 很关键，把预测的下一个词也要加入到idx中，用于预测next-next的token。
        idx = torch.cat([idx, idx_next], dim=1)
    return idx