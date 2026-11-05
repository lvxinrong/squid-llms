from transformers import AutoTokenizer
from squid.tokenizer.squid_tokenizer import SquidTokenizer


class QwenTokenizerAdapter(SquidTokenizer):
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer is None:
            raise RuntimeError(f"Tokenizer 加载失败：{model_name}")

        self.tokenizer = tokenizer

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, idx, **kwargs):
        return self.tokenizer.decode(idx, **kwargs)
