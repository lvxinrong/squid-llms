import torch
from squid.generate.engine import generate


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
        self.past_kv = None
        self.cached_token_ids = []
        self.next_position = 0

    def ask(self, user_input):
        prompt_ids = self._prompt_ids_within_budget(user_input)

        if self._cache_matches(prompt_ids):
            input_ids = prompt_ids[len(self.cached_token_ids):]
            past_kv = self.past_kv
        else:
            input_ids = self._slide_cache_to_prompt(prompt_ids)

            if input_ids is None:
                input_ids = prompt_ids
                past_kv = None
                self.next_position = 0
            else:
                past_kv = self.past_kv

        if not input_ids:
            input_ids = prompt_ids
            past_kv = None
            self.next_position = 0

        previous_cache_length = (
            past_kv[0][0].shape[2]
            if past_kv is not None
            else 0
        )

        position_offset = (
            self.next_position
            if getattr(
                self.model,
                "supports_position_offset",
                False,
            )
            else None
        )

        device = next(self.model.parameters()).device
        idx = torch.tensor(
            [input_ids],
            device=device,
            dtype=torch.long,
        )

        output, present_kv = generate(
            self.model,
            idx,
            max_new_tokens=self.max_new_tokens,
            context_size=self.context_size,
            eos_id=self.tokenizer.eos_id,
            past_kv=past_kv,
            return_past_kv=True,
            position_offset=position_offset,
        )

        generated_ids = output[0, len(input_ids):].tolist()
        response = self.tokenizer.decode(generated_ids)

        for marker in (
                "<|endoftext|>",
                "<|im_end|>",
                "### Instruction:",
        ):
            response = response.split(marker, 1)[0]

        response = response.strip()
        self.history.append((user_input, response))
        self.history = self.history[-self.max_history_turns:]

        full_token_ids = prompt_ids + generated_ids
        cache_length = present_kv[0][0].shape[2]

        self.cached_token_ids = full_token_ids[:cache_length]
        self.past_kv = present_kv

        processed_tokens = cache_length - previous_cache_length
        self.next_position += processed_tokens

        return response

    def clear(self):
        self.history = []
        self.past_kv = None
        self.cached_token_ids = []
        self.next_position = 0

    def _cache_matches(self, prompt_ids):
        cache_length = len(self.cached_token_ids)
        # 只有新 prompt 的开头，与 cache 已覆盖的 token 链完全一致，才允许复用
        return (
                self.past_kv is not None and 0 < cache_length <= len(prompt_ids) and prompt_ids[:cache_length] == self.cached_token_ids
        )

    """
    每次请求的时候，先计算提示词的存放空间 上下文长度 - 本轮对话最多生成的token数量 => 可以在上下文多少空间放提示词
    """
    def _prompt_ids_within_budget(self, user_input):
        prompt_budget = self.context_size - self.max_new_tokens

        if prompt_budget < 1:
            raise ValueError(
                "max_new_tokens 必须小于 context_size"
            )

        while True:
            # 构建新的prompt 包含历史对话和最新的输入
            prompt = self.prompt_builder(
                self.history,
                user_input,
            )
            prompt_ids = self.tokenizer.encode(prompt)

            # 如果新的提示词输入长度小于提示词可用空间，就直接返回最新的prompt
            if len(prompt_ids) <= prompt_budget:
                return prompt_ids

            if not self.history:
                raise ValueError(
                    "当前输入即使不带历史，也超过了 token 预算"
                )
            # 如果空间不够，也就是 len(prompt_ids) > prompt_budget 也就是上下文长度已经不足以放下提示词部分的时候，淘汰最早一轮。
            self.history.pop(0)

    # 将新 prompt 与旧 cache 对齐，保留可复用的前缀和最近后缀。
    def _slide_cache_to_prompt(self, prompt_ids):
        if not getattr(
                self.model,
                "supports_position_offset",
                False,
        ):
            return None

        if self.past_kv is None or not self.cached_token_ids:
            return None

        old_ids = self.cached_token_ids
        old_length = len(old_ids)

        prefix_length = 0
        prefix_limit = min(old_length, len(prompt_ids))

        while (
                prefix_length < prefix_limit
                and old_ids[prefix_length] == prompt_ids[prefix_length]
        ):
            prefix_length += 1

        if prefix_length == 0:
            return None

        max_suffix_length = min(
            old_length - prefix_length,
            len(prompt_ids) - prefix_length,
        )
        suffix_length = 0

        for candidate in range(max_suffix_length, 0, -1):
            if (
                    old_ids[old_length - candidate:]
                    == prompt_ids[
                prefix_length:prefix_length + candidate
            ]
            ):
                suffix_length = candidate
                break

        retained_ids = old_ids[:prefix_length]
        if suffix_length > 0:
            retained_ids += old_ids[-suffix_length:]

        present_kv = []
        for key, value in self.past_kv:
            if suffix_length > 0:
                key = torch.cat([
                    key[:, :, :prefix_length],
                    key[:, :, old_length - suffix_length:],
                ], dim=2)
                value = torch.cat([
                    value[:, :, :prefix_length],
                    value[:, :, old_length - suffix_length:],
                ], dim=2)
            else:
                key = key[:, :, :prefix_length].clone()
                value = value[:, :, :prefix_length].clone()

            present_kv.append((key, value))

        self.past_kv = present_kv
        self.cached_token_ids = retained_ids

        return prompt_ids[len(retained_ids):]


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
