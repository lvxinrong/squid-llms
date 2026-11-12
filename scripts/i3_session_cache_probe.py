"""I3-1 会话缓存探针：验证多轮复用、完整重算对照及第三轮滑动裁剪与位置进度。

用途：确认 token IDs、past_kv 和 next_position 对齐；第二轮只传新增 token，第三轮淘汰历史后仍正确复用部分 cache。
输入：真实 Qwen2.5-0.5B-Instruct 和固定三轮输入；报告写入 data/reports/i3-session-cache-probe.json。
边界：前两轮要求 cache / 重算回答一致；第三轮有意删除历史，回答允许变化，重点验收裁剪、增量输入和位置而非文本相等。
"""

import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

from squid.checkpoints.qwen import load_qwen2_5
from squid.device import get_device
from squid.generate.chat import ChatSession, build_qwen_chat_prompt
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
REPORT_PATH = ROOT / "data/reports/i3-session-cache-probe.json"
USER_INPUTS = [
    "用一句话解释 KV Cache。",
    "换成英文再说一遍。",
]


def ask_and_record(model, session, user_input):
    with patch.object(
        model,
        "forward",
        wraps=model.forward,
    ) as recorded_forward:
        response = session.ask(user_input)

    first_call = recorded_forward.call_args_list[0]
    return response, {
        "user_input": user_input,
        "response": response,
        "input_tokens": first_call.args[0].shape[1],
        "past_kv_used": first_call.kwargs["past_kv"] is not None,
        "past_cache_tokens": (
            first_call.kwargs["past_kv"][0][0].shape[2]
            if first_call.kwargs["past_kv"] is not None
            else 0
        ),
        "position_offset": first_call.kwargs.get("position_offset"),
        "history_turns": len(session.history),
        "cached_tokens": len(session.cached_token_ids),
    }


def main():
    device = get_device()
    context_size = 4096
    model = load_qwen2_5(
        WEIGHTS_PATH,
        cfg={"context_length": context_size},
        device=device,
    )
    tokenizer = QwenTokenizerAdapter("Qwen/Qwen2.5-0.5B-Instruct")
    prompt_builder = partial(
        build_qwen_chat_prompt,
        tokenizer.tokenizer,
    )

    cached_session = ChatSession(
        model,
        tokenizer,
        max_new_tokens=16,
        context_size=context_size,
        prompt_builder=prompt_builder,
    )
    rebuild_session = ChatSession(
        model,
        tokenizer,
        max_new_tokens=16,
        context_size=context_size,
        prompt_builder=prompt_builder,
    )

    turns = []
    for user_input in USER_INPUTS:
        cached_response, cached = ask_and_record(
            model,
            cached_session,
            user_input,
        )

        rebuild_session.past_kv = None
        rebuild_session.cached_token_ids = []
        rebuild_response, rebuild = ask_and_record(
            model,
            rebuild_session,
            user_input,
        )

        turns.append({
            "cached": cached,
            "full_rebuild": rebuild,
            "responses_match": cached_response == rebuild_response,
        })

    sliding_user_input = "请用一句话总结我们今天讨论的主题。"
    retained_history = cached_session.history[1:]
    sliding_prompt_ids = tokenizer.encode(
        prompt_builder(retained_history, sliding_user_input),
    )
    cached_session.context_size = len(sliding_prompt_ids) + 16
    cache_tokens_before_eviction = cached_session.past_kv[0][0].shape[2]
    next_position_before = cached_session.next_position

    _, sliding = ask_and_record(
        model,
        cached_session,
        sliding_user_input,
    )
    sliding.update({
        "cache_tokens_before_eviction": cache_tokens_before_eviction,
        "next_position_before": next_position_before,
        "next_position_after": cached_session.next_position,
        "prompt_tokens_after_eviction": len(sliding_prompt_ids),
        "context_size_after_eviction": cached_session.context_size,
        "cache_was_truncated": (
            0 < sliding["past_cache_tokens"] < cache_tokens_before_eviction
        ),
        "input_is_incremental": (
            sliding["input_tokens"] < len(sliding_prompt_ids)
        ),
        "position_preserved": (
            sliding["position_offset"] == next_position_before
        ),
        "next_position_correct": (
            cached_session.next_position
            == next_position_before
            + sliding["cached_tokens"]
            - sliding["past_cache_tokens"]
        ),
        "history_evicted": [
            user for user, _ in cached_session.history
        ] == [USER_INPUTS[1], sliding_user_input],
        "response_equality_checked": False,
    })

    report = {
        "experiment": "i3-session-cache-probe",
        "model": "Qwen2.5-0.5B-Instruct",
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "max_new_tokens": 16,
        "context_size": context_size,
        "turns": turns,
        "all_responses_match": all(
            turn["responses_match"] for turn in turns
        ),
        "second_turn_cache_reused": turns[1]["cached"]["past_kv_used"],
        "sliding_turn": sliding,
        "sliding_checks_passed": all([
            sliding["cache_was_truncated"],
            sliding["input_is_incremental"],
            sliding["position_preserved"],
            sliding["next_position_correct"],
            sliding["history_evicted"],
        ]),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    assert report["all_responses_match"], "会话 cache 与完整重算的回答不一致"
    assert report["second_turn_cache_reused"], "第二轮没有复用会话 cache"
    assert report["sliding_checks_passed"], "真实 Qwen 滑动 cache 验收失败"
    print(f"\n会话 cache 探针报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()
