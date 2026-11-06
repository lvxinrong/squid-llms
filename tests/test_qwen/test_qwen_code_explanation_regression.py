from functools import partial
from pathlib import Path

import torch

from squid.checkpoints.qwen import load_qwen2_5
from squid.generate.chat import ChatSession, build_qwen_chat_prompt
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def test_qwen_code_explanation_regression():
    model = load_qwen2_5(
        ROOT / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors",
        device=torch.device("cpu"),
    )
    tokenizer = QwenTokenizerAdapter(MODEL_NAME)
    # partial 固定一个函数的部分参数，生成一个新的可调用对象（partial object）。调用这个新对象时，只需传剩下的参数。
    """
    from functools import partial
    def power(base, exp):
        return base ** exp

    square = partial(power, exp=2)
    cube   = partial(power, exp=3)

    print(square(5))  # 25
    print(cube(2))    # 8
    """
    prompt_builder = partial(
        build_qwen_chat_prompt,
        tokenizer.tokenizer,
    )

    session = ChatSession(
        model,
        tokenizer,
        max_new_tokens=512,
        context_size=1024,
        prompt_builder=prompt_builder,
    )

    messages = [
        "你好，我叫吕炘嵘。",
        "我喜欢写代码，也正在学习大模型。",
        (
            "请根据上面的对话回答："
            "用户叫什么名字？用户喜欢做什么？正在学习什么？"
            "不要介绍你自己。"
        ),
    ]

    response = session.ask(
        'def find_even_numbers(numbers):\n'
        '    even_numbers = []\n\n'
        '    for number in numbers:\n'
        '        if number % 2 == 0:\n'
        '            even_numbers.append(number)\n\n'
        '    return even_numbers\n\n\n'
        'numbers = [1, 2, 3, 4, 5, 6]\n'
        'result = find_even_numbers(numbers)\n\n'
        'print("偶数列表：", result)\n\n'
        '请逐行解释这段 Python 代码，说明函数的执行过程，'
        '并告诉我最终会输出什么。'
    )

    print(response)

    assert "find_even_numbers" in response
    assert "偶数" in response
    assert "偶数列表： [2, 4, 6]" in response

    assert "<|im_start|>" not in response
    assert "<|im_end|>" not in response
    assert "### Instruction:" not in response

