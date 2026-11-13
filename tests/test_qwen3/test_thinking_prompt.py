from squid.generate.chat import build_qwen_chat_prompt


class StubTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "prompt"


def test_qwen_prompt_preserves_default_template_behavior():
    tokenizer = StubTokenizer()
    build_qwen_chat_prompt(tokenizer, [], "你好")
    assert "enable_thinking" not in tokenizer.kwargs


def test_qwen_prompt_can_enable_and_disable_thinking():
    for enabled in (True, False):
        tokenizer = StubTokenizer()
        build_qwen_chat_prompt(
            tokenizer,
            [],
            "你好",
            enable_thinking=enabled,
        )
        assert tokenizer.kwargs["enable_thinking"] is enabled
