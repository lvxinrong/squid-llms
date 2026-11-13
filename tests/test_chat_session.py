from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from squid.generate.chat import ChatSession
from squid.model.v1_gpt.gpt import GPTModel
from squid.model.v3_modern.gpt import GPTModelV3

TEST_CFG = {
    "vocab_size": 128,
    "emb_dim": 16,
    "context_length": 512,
    "n_layers": 2,
    "n_heads": 4,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

class AsciiTokenizer:
    eos_id = None

    def encode(self, text):
        return [
            0 if char == "A" else ord(char) + 1
            for char in text
        ]

    def decode(self, idx):
        return "".join(
            "A" if token_id == 0 else chr(token_id - 1)
            for token_id in idx
        )

def test_second_turn_only_processes_new_tokens():
    torch.manual_seed(123)
    model = GPTModel(TEST_CFG).eval()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    tokenizer = AsciiTokenizer()

    session = ChatSession(
        model,
        tokenizer,
        max_new_tokens = 3,
        context_size = 512,
    )

    assert session.ask("one") == "AAA"

    previous_cache = session.past_kv
    previous_cached_ids = session.cached_token_ids.copy()

    second_prompt = session.prompt_builder(
        session.history,
        "two",
    )

    second_prompt_ids = tokenizer.encode(second_prompt)

    assert session._cache_matches(second_prompt_ids)

    with patch.object(
        model, "forward", wraps = model.forward
    ) as recorded_forward:
        assert session.ask("two") == "AAA"

    first_call = recorded_forward.call_args_list[0]
    incremental_idx = first_call.args[0]

    assert first_call.kwargs["past_kv"] is previous_cache

    assert incremental_idx.shape[1] == (
        len(second_prompt_ids) - len(previous_cached_ids)
    )


def build_session(context_size):
    torch.manual_seed(123)
    model = GPTModel(TEST_CFG).eval()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    return ChatSession(
        model,
        AsciiTokenizer(),
        max_new_tokens=3,
        context_size=context_size,
    )


def test_prompt_budget_evicts_oldest_turns_and_rebuilds_cache():
    session = build_session(context_size=512)
    tokenizer = session.tokenizer
    base_prompt_ids = tokenizer.encode(
        session.prompt_builder([], "new"),
    )
    session.context_size = len(base_prompt_ids) + 3

    assert session.ask("one") == "AAA"
    session.history.extend([
        ("old two", "AAA"),
        ("old three", "AAA"),
    ])

    with patch.object(
        session.model,
        "forward",
        wraps=session.model.forward,
    ) as recorded_forward:
        assert session.ask("new") == "AAA"

    assert session.history == [("new", "AAA")]

    first_call = recorded_forward.call_args_list[0]
    assert first_call.kwargs["past_kv"] is None
    assert first_call.args[0].shape[1] == len(base_prompt_ids)


def test_prompt_budget_rejects_oversized_current_input():
    session = build_session(context_size=512)
    prompt_ids = session.tokenizer.encode(
        session.prompt_builder([], "too long"),
    )
    session.context_size = len(prompt_ids) + 2

    with pytest.raises(ValueError, match="不带历史"):
        session.ask("too long")

    assert session.history == []
    assert session.past_kv is None
    assert session.cached_token_ids == []


def test_slide_cache_preserves_exact_prefix_and_suffix():
    session = ChatSession(
        SimpleNamespace(supports_position_offset=True),
        AsciiTokenizer(),
    )
    key = torch.arange(6).reshape(1, 1, 6, 1)
    value = key + 100
    session.past_kv = [(key, value)]
    session.cached_token_ids = [10, 11, 12, 20, 21, 22]
    session.next_position = 6

    input_ids = session._slide_cache_to_prompt(
        [10, 11, 20, 21, 22, 30, 31],
    )

    assert input_ids == [30, 31]
    assert session.cached_token_ids == [10, 11, 20, 21, 22]
    assert session.past_kv[0][0].flatten().tolist() == [0, 1, 3, 4, 5]
    assert session.past_kv[0][1].flatten().tolist() == [100, 101, 103, 104, 105]
    assert session.next_position == 6


V3_TEST_CFG = {
    "vocab_size": 128,
    "emb_dim": 16,
    "ffn_dim": 32,
    "n_heads": 4,
    "n_kv_heads": 2,
    "n_layers": 2,
    "rope_theta": 10000.0,
    "eps": 1e-6,
    "qkv_bias": True,
    "ffn_bias": False,
    "tie_word_embeddings": False,
    "drop_rate": 0.0,
    "context_length": 512,
}


def test_v3_sliding_cache_preserves_position_progress():
    torch.manual_seed(123)
    model = GPTModelV3(V3_TEST_CFG).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    tokenizer = AsciiTokenizer()
    session = ChatSession(
        model,
        tokenizer,
        max_new_tokens=3,
        context_size=512,
    )

    assert session.ask("one") == "AAA"
    assert session.ask("two") == "AAA"

    old_cache_length = session.past_kv[0][0].shape[2]
    old_next_position = session.next_position
    retained_history = session.history[1:]
    expected_prompt_ids = tokenizer.encode(
        session.prompt_builder(retained_history, "three"),
    )
    session.context_size = len(expected_prompt_ids) + 3

    with patch.object(
        model,
        "forward",
        wraps=model.forward,
    ) as recorded_forward:
        assert session.ask("three") == "AAA"

    calls = recorded_forward.call_args_list
    first_call = calls[0]
    retained_cache = first_call.kwargs["past_kv"]
    retained_length = retained_cache[0][0].shape[2]
    input_length = first_call.args[0].shape[1]

    assert retained_length > 0
    assert retained_length < old_cache_length
    assert input_length < len(expected_prompt_ids)
    assert first_call.kwargs["position_offset"] == old_next_position
    assert calls[1].kwargs["position_offset"] == old_next_position + input_length

    assert session.history == [("two", "AAA"), ("three", "AAA")]
    assert session.cached_token_ids == expected_prompt_ids + [0, 0, 0]
    assert session.next_position == old_next_position + input_length + 3


def test_ask_stream_matches_ask_and_preserves_state():
    synchronous = build_session(context_size=512)
    streaming = build_session(context_size=512)

    expected = synchronous.ask("one")
    deltas = list(streaming.ask_stream("one"))

    assert "".join(deltas) == expected
    assert streaming.history == synchronous.history
    assert streaming.cached_token_ids == synchronous.cached_token_ids
    assert streaming.next_position == synchronous.next_position
    assert streaming.last_generated_tokens == 3
    assert streaming.last_finish_reason == "length"

    for stream_layer, sync_layer in zip(
        streaming.past_kv,
        synchronous.past_kv,
    ):
        for stream_tensor, sync_tensor in zip(stream_layer, sync_layer):
            assert torch.equal(stream_tensor, sync_tensor)


def test_ask_stream_waits_for_incomplete_utf8_suffix():
    class IncompleteUtf8Tokenizer(AsciiTokenizer):
        def decode(self, idx):
            if idx == [1]:
                return ""
            if idx == [1, 2]:
                return "总结"
            return super().decode(idx)

    session = build_session(context_size=512)
    session.tokenizer = IncompleteUtf8Tokenizer()

    def fake_generate_stream(model, idx, **kwargs):
        yield torch.tensor(1)
        yield torch.tensor(2)
        cache = torch.zeros(1, 1, idx.shape[1] + 2, 1)
        return None, [(cache, cache)]

    with patch("squid.generate.chat.generate_stream", new=fake_generate_stream):
        deltas = list(session.ask_stream("one"))

    assert deltas == ["总结"]
    assert session.history == [("one", "总结")]


def test_incomplete_stream_invalidates_cache_and_can_recover():
    session = build_session(context_size=512)
    stream = session.ask_stream("one")

    assert next(stream) == "A"
    stream.close()

    assert session.history == []
    assert session.past_kv is None
    assert session.cached_token_ids == []
    assert session.next_position == 0

    assert session.ask("one") == "AAA"