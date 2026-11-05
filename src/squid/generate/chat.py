from squid.generate.text import generate_text


class ChatSession:
    def __init__(
        self,
        model,
        tokenizer,
        max_history_turns=5,
        max_new_tokens=256,
        context_size=1024,
        prompt_builder=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.history = []
        self.max_history_turns = max_history_turns
        self.max_new_tokens = max_new_tokens
        self.context_size = context_size
        self.prompt_builder = prompt_builder or build_chat_prompt


    def ask(self, user_input):
        prompt = self.prompt_builder(self.history, user_input)
        responses = generate_text(self.model,
                                  self.tokenizer,
                                  prompt,
                                  max_new_tokens=self.max_new_tokens,
                                  context_size=self.context_size,
                                  eos_id=self.tokenizer.eos_id)
        response = responses[len(prompt):]
        for marker in ("<|endoftext|>", "<|im_end|>", "### Instruction:"):
            response = response.split(marker, 1)[0]
        response = response.strip()
        self.history.append((user_input, response))
        self.history = self.history[-self.max_history_turns:]
        # 截取最新的回答
        return response

    def clear(self):
        self.history = []


def build_chat_prompt(history, current_input):
    parts = [
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    ]

    for user_input, assistant_response in history:
        parts.append(
            f"### Instruction:\n{user_input}\n\n"
            f"### Response:\n{assistant_response}"
        )

    parts.append(
        f"### Instruction:\n{current_input}\n\n"
        f"### Response:\n"
    )
    return "\n\n".join(parts)


def build_qwen_chat_prompt(hf_tokenizer, history, current_input):
    messages = [
        {
            "role": "system",
            "content": "你是一个乐于助人的 AI 助手。",
        }
    ]

    for user_input, assistant_response in history:
        messages.append({
            "role": "user",
            "content": user_input,
        })
        messages.append({
            "role": "assistant",
            "content": assistant_response,
        })

    messages.append({
        "role": "user",
        "content": current_input,
    })

    return hf_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
