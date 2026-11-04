import torch

from squid.generate.engine import generate
from squid.tokenizer import TiktokenTokenizer
from squid.tokenizer.squid_tokenizer import SquidTokenizer

GEN_TEXT_DEFAULT_TOKENIZER = TiktokenTokenizer("gpt2")

def generate_text(model, tokenizer: SquidTokenizer, prompt: str, **kwargs) -> str:
    if tokenizer is None:
        tokenizer = GEN_TEXT_DEFAULT_TOKENIZER
    device = next(model.parameters()).device
    idx= torch.tensor([tokenizer.encode(prompt)]).to(device)
    out = generate(model, idx, **kwargs)
    return tokenizer.decode(out[0].tolist())