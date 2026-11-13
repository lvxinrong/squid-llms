import time

import torch
from squid.generate.engine import generate_no_cache, sample_next
from squid.generate.benchmark import synchronize

"""
给定 Draft 候选和 Target 对应位置的贪心答案，计算本轮结果
"""
def verify_greedy_draft(draft_ids, target_ids):
    if not draft_ids:
        raise ValueError("draft_ids 不能为空")

    if len(target_ids) != len(draft_ids) + 1:
        raise ValueError("target_ids 必须比 draft_ids 多一个 bonus token")

    output_ids = []
    accepted_draft_count = 0
    used_target_correction = False

    for index, draft_id in enumerate(draft_ids):
        target_id = target_ids[index]

        if draft_id == target_id:
            output_ids.append(draft_id)
            accepted_draft_count += 1
            continue

        output_ids.append(target_id)
        used_target_correction = True
        break

    if accepted_draft_count == len(draft_ids):
        output_ids.append(target_ids[-1])

    return {
        "output_ids": output_ids,
        "accepted_draft_count": accepted_draft_count,
        "rejected_draft_count": (
                len(draft_ids) - accepted_draft_count
        ),
        "used_target_correction": used_target_correction,
    }

def generate_speculative_no_cache(
        draft_model,
        target_model,
        idx,
        max_new_tokens,
        context_size,
        draft_lookahead = 4,):

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须大于等于 1")
    if draft_lookahead < 1:
        raise ValueError("draft_lookahead 必须大于等于 1")

    original_length = idx.shape[1]

    while idx.shape[1] - original_length < max_new_tokens:
        remaining = max_new_tokens - (idx.shape[1] - original_length)
        lookahead = min(draft_lookahead, remaining)

        draft_output = generate_no_cache(draft_model, idx, max_new_tokens = lookahead,
                                         context_size = context_size,
                                         temperature = 0.0,
                                         eos_id = None)

        draft_ids = draft_output[0, idx.shape[1]:,].tolist()

        with torch.no_grad():
            target_logits = target_model(draft_output)

        current_length = idx.shape[1]
        target_start = current_length - 1
        target_end = target_start + lookahead + 1

        target_ids = target_logits[0, target_start : target_end, :,].argmax(dim = -1).tolist()

        verification = verify_greedy_draft(draft_ids, target_ids)

        output_ids = verification["output_ids"][:remaining]
        output = torch.tensor(
            [output_ids],
            dtype=torch.long,
            device=idx.device,
        )
        idx = torch.cat([idx, output], dim=1)

    return idx


def truncate_past_kv(past_kv, length):
    current_length = past_kv[0][0].shape[2]

    if length < 0 or length > current_length:
        raise ValueError(
            "cache 裁剪长度必须位于 0 与当前长度之间"
        )

    return [
        (
            key[:, :, :length].clone(), # clone() 是为了让新 cache 拥有独立存储，避免切片视图继续引用完整旧 cache，导致被拒绝部分无法释放。
            value[:, :, :length].clone(),
        )
        for key, value in past_kv
    ]


"""
整体流程介绍如下:
开始:
    Draft prefill：得到 draft_logits / draft_past
    Target prefill：得到 target_logits / target_past

Draft 逐 token 猜 k 个候选

Target 一次性校验 k 个候选

verify_greedy_draft 计算接受、修正和拒绝:
    有拒绝 → 双方 cache 回滚
    无拒绝 → 保留全部草稿 cache
    
将修正 / bonus token 同步输入双方模型

进入下一轮，直到达到 max_new_tokens

Bonus token 就是：当本轮 k 个 Draft token 全部被 Target 接受后，Target 顺便多生成的一个 token。
“Bonus”可以理解为“额外赠送”，它不是特殊 token，也不是奖励机制，只是一个普通的下一个 token。

Target 处理完第 4 个 Draft token 后，第 4 个位置输出的 logits，用来预测第 5 个 token，也就是 bonus token。

"""
def generate_speculative_cached(
    draft_model,
    target_model,
    idx,
    max_new_tokens,
    context_size,
    draft_lookahead=4,
    return_stats=False,
):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须大于等于 1")
    if draft_lookahead < 1:
        raise ValueError("draft_lookahead 必须大于等于 1")
    if idx.shape[1] + max_new_tokens > context_size:
        raise ValueError(
            "当前 cached 实验要求 prompt 与生成结果不超过 context_size"
        )

    original_length = idx.shape[1]
    device = idx.device
    started_at = time.perf_counter()

    stats = {
        "rounds": 0,
        "draft_tokens_proposed": 0,
        "draft_tokens_accepted": 0,
        "draft_tokens_rejected": 0,
        "correction_rounds": 0,
        "bonus_tokens": 0,
        "ttft_seconds": None,
    }

    # 开始前：Draft 和 Target 各自 prefill
    """
    同一份 prompt
    ├── Draft prefill
    │   → draft_logits：预测 Draft 的第一个候选 token
    │   → draft_past：Draft 自己的 prompt cache
    │
    └── Target prefill
        → target_logits：Target 对下一个 token 的预测
        → target_past：Target 自己的 prompt cache
    """
    with torch.no_grad():
        draft_logits, draft_past = draft_model(
            idx,
            use_cache=True,
        )
        target_logits, target_past = target_model(
            idx,
            use_cache=True,
        )


    while idx.shape[1] - original_length < max_new_tokens:
        generated_count = idx.shape[1] - original_length
        remaining = max_new_tokens - generated_count
        lookahead = min(draft_lookahead, remaining)

        base_cache_length = target_past[0][0].shape[2]
        draft_cache_length = draft_past[0][0].shape[2]

        if draft_cache_length != base_cache_length:
            raise RuntimeError("Draft 与 Target 的 cache 长度不一致")

        draft_ids = []
        # Draft 阶段：连续猜 k 个 token
        # 采出 A → 模型处理 A → 得到新 logits
        # 采出 B → 模型处理 B → 得到新 logits
        # 采出 C → 模型处理 C → 得到新 logits
        # 采出 D → 模型处理 D → 得到新 logits
        for _ in range(lookahead):
            next_id = sample_next(
                draft_logits[:, -1, :],
                temperature=0.0,
            )
            draft_ids.append(next_id.item())

            with torch.no_grad():
                draft_logits, draft_past = draft_model(
                    next_id,
                    past_kv=draft_past,
                    use_cache=True,
                )

        draft_idx = torch.tensor(
            [draft_ids],
            dtype=torch.long,
            device=device,
        )

        # Target 阶段：一次校验整段草稿, 这里只调用 Target 一次，但输入包含 4 个 token。
        """
        这就是多 token 增量 prefill：
            Target 原有 cache
            +
            A、B、C、D 四个候选
            → 一次前向
            → 得到每个候选之后的预测
            → 同时得到包含这四个候选的新 Target cache
        """
        with torch.no_grad():
            verification_logits, target_past = target_model(
                draft_idx,
                past_kv=target_past,
                use_cache=True,
            )

        # 构造 Target 的答案序列
        target_ids = [
            # target_logits：Target 在看到草稿之前，对第一个草稿 token 的答案；
            target_logits[0, -1, :].argmax().item()
        ]
        target_ids.extend(
            # verification_logits：Target 在处理 A、B、C、D 后，分别对下一个 token 的答案。
            verification_logits[0].argmax(dim=-1).tolist()
        )
        """
        纯函数负责决定接受、修正与拒绝
        Draft 的 A、B、C、D  vs  Target 对每个位置的答案
        """
        verification = verify_greedy_draft(
            draft_ids,
            target_ids,
        )

        stats["rounds"] += 1
        stats["draft_tokens_proposed"] += len(draft_ids)
        stats["draft_tokens_accepted"] += (
            verification["accepted_draft_count"]
        )
        stats["draft_tokens_rejected"] += (
            verification["rejected_draft_count"]
        )

        if verification["used_target_correction"]:
            stats["correction_rounds"] += 1

        accepted_count = verification["accepted_draft_count"]

        if verification["rejected_draft_count"] > 0:
            # Target 校验后，cache 里已经有被拒绝内容 Target 刚才一次性处理了 A、B、C、D，所以它的 cache 已经包含它们。
            retained_length = base_cache_length + accepted_count
            """
            举例: 
            假设 Draft： A   B   C   D
                Target：A   B   X   ...
                结果： 接受 接受 修正 丢弃
            draft的C,D 被拒绝了就需要回滚处理，并且draft和target的past_kv 都需要回滚
            """
            target_past = truncate_past_kv(
                target_past,
                retained_length,
            )
            draft_past = truncate_past_kv(
                draft_past,
                retained_length,
            )

        output_ids = verification["output_ids"][:remaining]

        if (
                not verification["used_target_correction"]
                and len(output_ids) > accepted_count
        ):
            stats["bonus_tokens"] += 1

        output = torch.tensor(
            [output_ids],
            dtype=torch.long,
            device=device,
        )
        idx = torch.cat([idx, output], dim=1)

        if return_stats and stats["ttft_seconds"] is None:
            synchronize(idx.device)
            stats["ttft_seconds"] = (
                    time.perf_counter() - started_at
            )

        generated_count = idx.shape[1] - original_length
        if generated_count >= max_new_tokens:
            break

        # 修正 token 或 bonus token 也必须进入双方模型。
        """
        Target：把 X 的 K/V 加入 Target cache，并得到下一次预测
        Draft： 把 X 的 K/V 加入 Draft cache，并得到下一次预测
        """
        sync_id = verification["output_ids"][-1]
        sync_idx = torch.tensor(
            [[sync_id]],
            dtype=torch.long,
            device=device,
        )
        # 这样下一轮开始时，双方才会面对同一份真实历史。
        with torch.no_grad():
            target_logits, target_past = target_model(
                sync_idx,
                past_kv=target_past,
                use_cache=True,
            )
            draft_logits, draft_past = draft_model(
                sync_idx,
                past_kv=draft_past,
                use_cache=True,
            )

    if return_stats:
        return idx, stats

    return idx


