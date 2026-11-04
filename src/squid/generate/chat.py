from squid.generate.text import generate_text


class ChatSession:
    def __init__(self, model, tokenizer, max_history_turns=5, max_new_tokens=256, context_size=1024,):
        self.model = model
        self.tokenizer = tokenizer
        self.history = []
        self.max_history_turns = max_history_turns
        self.max_new_tokens = max_new_tokens
        self.context_size = context_size

    def ask(self, user_input):
        prompt = build_chat_prompt(self.history, user_input)
        responses = generate_text(self.model,
                                  self.tokenizer,
                                  prompt,
                                  max_new_tokens=self.max_new_tokens,
                                  context_size=self.context_size,
                                  eos_id=self.tokenizer.eos_id)
        # 处理history
        response = responses[len(prompt):].replace("<|endoftext|>", "").strip()
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
