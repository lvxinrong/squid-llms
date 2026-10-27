from squid.tokenizer.squid_tokenizer import SquidTokenizer
import tiktoken

class TiktokenTokenizer(SquidTokenizer):
    def __init__(self, encoding_name:str) -> None:
        self._tok = tiktoken.get_encoding(encoding_name)

    """
    t.encode(text)                                        # 默认：遇到特殊token就炸
    t.encode(text, allow_special=True)                    # 全部放行，<|endoftext|> → 50256
    t._tok.encode(text, allowed_special={"<|endoftext|>"}) # 只放行这一个，其他特殊token照炸
    """
    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        return self._tok.encode(text, allowed_special="all" if allow_special else set())

    def decode(self, idx: list[int]) -> str:
        return self._tok.decode(idx)

    @property
    def eos_id(self) -> int:
        return self._tok.eot_token