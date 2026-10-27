from squid.tokenizer import SquidTokenizer, bpe

class MyBPETokenizer(SquidTokenizer):
    """我自己手写的bpe"""
    def __init__(self, vocab, merges, eos_id=None):
        self.vocab = vocab
        self.merges = merges
        self._eos_id = eos_id if eos_id is not None else len(vocab)
        self.vocab.setdefault(self._eos_id, b"<|endoftext|>")

    def encode(self, text: str) -> list[int]:
        return bpe.encode(text, self.merges)

    def decode(self, idx: list[int]) -> str:
        return bpe.decode(idx, self.vocab)

    @classmethod
    def from_directory(cls, path):
        vocab, merges = bpe.load_bpe(path)
        return cls(vocab, merges)

    @property
    def eos_id(self) -> int:
        return self._eos_id