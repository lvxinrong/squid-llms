import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from squid.device import get_device
from squid.generate.chat import build_qwen_chat_prompt

def test_qwen_2():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    history = [
        (
            "你好，我叫吕炘嵘。",
            "你好！很高兴见到你。有什么我可以帮助你的吗？",
        ),
        (
            "我喜欢写代码，也正在学习大模型。",
            "那太好了！编程和大模型都是非常有趣和有挑战性的领域。"
            "你对编程有什么具体的问题或项目吗？",
        ),
    ]

    current_input = (
        "请根据上面的对话回答："
        "用户叫什么名字？用户喜欢做什么？正在学习什么？"
        "不要介绍你自己。"
    )

    prompt = build_qwen_chat_prompt(
        tokenizer,
        history,
        current_input,
    )

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    ).to(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    ).to(device)
    model.eval()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    print(response)
