import os

from openai import OpenAI

from squid.env import load_env


BASE_URL = "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1"
_client = None


def get_client():
    global _client
    if _client is None:
        load_env()
        api_key = os.environ.get("QWEN_API_KEY")
        if not api_key:
            raise RuntimeError(
                "缺少 QWEN_API_KEY：在项目根目录 .env 里加一行 "
                "QWEN_API_KEY=sk-xxx（可参考 .env.example），"
                "或在运行环境里配置同名环境变量。"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
        )
    return _client


def query_judge(prompt, model="qwen4-max", max_tokens=256):
    resp = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    usage = resp.usage
    result = {
        "content": choice.message.content,
        "finish_reason": choice.finish_reason,
        "model": resp.model,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
    }

    reasoning_content = getattr(
        choice.message,
        "reasoning_content",
        None,
    )
    if reasoning_content is not None:
        result["reasoning_content"] = reasoning_content

    details = (
        getattr(usage, "completion_tokens_details", None)
        if usage
        else None
    )
    reasoning_tokens = (
        getattr(details, "reasoning_tokens", None)
        if details
        else None
    )
    if reasoning_tokens is not None:
        result["usage"]["reasoning_tokens"] = reasoning_tokens

    return result
