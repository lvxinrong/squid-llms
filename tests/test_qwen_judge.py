from types import SimpleNamespace

import pytest

from squid.eval import qwen_judge


@pytest.fixture(autouse=True)
def isolate_qwen_client(monkeypatch):
    monkeypatch.setattr(qwen_judge, "_client", None)
    monkeypatch.setattr(qwen_judge, "load_env", lambda *a, **k: False)


def test_missing_qwen_api_key_raises(monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="QWEN_API_KEY"):
        qwen_judge.get_client()


def test_qwen_query_uses_token_plan_contract(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "test-qwen-key")
    calls = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="A",
                    reasoning_content="测试推理",
                ),
                finish_reason="stop",
            )
        ],
        model="qwen4-max",
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=6,
            total_tokens=26,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=5
            ),
        ),
    )

    def create(**kwargs):
        calls["request"] = kwargs
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )

    def openai_client(**kwargs):
        calls["client"] = kwargs
        return client

    monkeypatch.setattr(qwen_judge, "OpenAI", openai_client)

    result = qwen_judge.query_judge("比较两条回答")

    assert calls["client"] == {
        "api_key": "test-qwen-key",
        "base_url": (
            "https://token-plan.maas.qianwenaiapi.com/"
            "compatible-mode/v1"
        ),
    }
    assert calls["request"] == {
        "model": "qwen4-max",
        "messages": [{"role": "user", "content": "比较两条回答"}],
        "temperature": 0,
        "max_tokens": 256,
    }
    assert result == {
        "content": "A",
        "finish_reason": "stop",
        "model": "qwen4-max",
        "reasoning_content": "测试推理",
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 6,
            "total_tokens": 26,
            "reasoning_tokens": 5,
        },
    }

def test_qwen_query_allows_max_tokens_override(monkeypatch):
    calls = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="tie"),
                finish_reason="stop",
            )
        ],
        model="qwen4-max",
        usage=None,
    )

    def create(**kwargs):
        calls.update(kwargs)
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    monkeypatch.setattr(qwen_judge, "get_client", lambda: client)

    result = qwen_judge.query_judge(
        "比较两条回答",
        max_tokens=512,
    )

    assert calls["max_tokens"] == 512
    assert result["content"] == "tie"
    assert result["finish_reason"] == "stop"

