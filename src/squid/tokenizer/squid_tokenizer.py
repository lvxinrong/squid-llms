from typing import Protocol


class SquidTokenizer(Protocol):
    """分词器协议：encode/decode 的最小契约。"""
    def encode(self, text: str) -> list[int]:...
    def decode(self, idx: list[int]) -> str:...

    """
    在 Protocol 里写 @property，作用是声明“实现类必须提供一个名为 eos_id 的只读属性”；使用它时写成 tok.eos_id（不加括号），而不是 tok.eos_id()。
    """
    @property
    def eos_id(self) -> int:...