import json
import os
import re

from tqdm import tqdm
from openai import OpenAI

from squid.data.instruct import format_input
from squid.env import load_env

_client = None


def get_client():
    # 懒加载：import 本模块不需要 key，真正调用时才读 .env / 环境变量
    global _client
    if _client is None:
        load_env()  # 项目根目录 .env 兜底；已存在的环境变量优先
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "缺少 DEEPSEEK_API_KEY：在项目根目录 .env 里加一行 "
                "DEEPSEEK_API_KEY=sk-xxx（可参考 .env.example），"
                "或在运行环境里配置同名环境变量。"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",  # 不要带 /chat/completions
        )
    return _client


def dump_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def score_entries(json_data, json_key="model_response",
                  save_path="instruction-data-with-scores.json",
                  flush_every=10):
    for i, entry in enumerate(tqdm(json_data, desc="Scoring entries"), 1):
        if entry.get("score") is not None:
            continue  # 已打过分，断点续跑不重复烧额度

        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )

        result = query_judge(prompt)
        entry["judge"] = result                       # 完整原始信息
        entry["score"] = parse_score(result["content"])

        if entry["score"] is None:
            print(f"[{i}] 解析失败，原始回复: {result['content']!r}")

        if i % flush_every == 0:
            dump_json(json_data, save_path)

    dump_json(json_data, save_path)
    return json_data


def query_judge(prompt, model="deepseek-chat"):
    resp = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=16,
    )
    choice = resp.choices[0]
    usage = resp.usage
    return {
        "content": choice.message.content,
        "finish_reason": choice.finish_reason,
        "model": resp.model,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
    }


def parse_score(text):
    """把模型回复解析成 0-100 的整数；解析不出来返回 None。"""
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    return max(0, min(100, int(match.group())))


def summarize_scores(data):
    scores = sorted(e["score"] for e in data if isinstance(e.get("score"), int))
    if not scores:
        print("没有有效打分")
        return
    print(f"有效打分 {len(scores)}/{len(data)}  |  "
          f"mean={sum(scores) / len(scores):.1f}  median={scores[len(scores) // 2]}  "
          f"min={scores[0]}  max={scores[-1]}  "
          f"p10={scores[int(.1 * len(scores))]}  p90={scores[int(.9 * len(scores))]}")
    for lo, hi in [(0, 0), (1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]:
        n = sum(1 for s in scores if lo <= s <= hi)
        print(f"{lo:3d}-{hi:3d}: {n:3d}  {'█' * int(60 * n / len(scores))}")


if __name__ == '__main__':
    with open("instruction-data-with-response.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    data = score_entries(json_data)
    summarize_scores(data)
