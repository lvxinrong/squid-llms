"""用 DeepSeek API 将脚本内置的英文模型输出生成自然中文，用于分享与展示。

用途：把翻译调用与模型训练解耦，单独检查英文文本的中文表达效果。
输入：脚本中的 EN_TEXT 和环境变量 DEEPSEEK_API_KEY；结果直接打印到终端。
边界：这是外部 API 翻译，不是本地 Squid 模型的翻译能力验收，也不产出可复现的本地报告。
"""
import os

from openai import OpenAI
from squid.env import load_env

load_env()
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

EN_TEXT = """The meaning of life is to live in the present. Living in the present is not the same thing as living in the past: the past is always changing and the future can be different as well. When it comes to understanding the meaning of life, it is important to understand how this new future will affect us."""

prompt = (
    "把下面的英文翻译成自然、流畅、有文学感的中文。"
    "要求：信达雅，不要逐字硬翻，保留原文的气质，适合发朋友圈。\n\n"
    f"{EN_TEXT}"
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
)
print(resp.choices[0].message.content)
