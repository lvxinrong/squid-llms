import torch

from squid.generate.engine import generate
from squid.tokenizer.squid_tokenizer import SquidTokenizer


def generate_text(model, tokenizer: SquidTokenizer, prompt: str, **kwargs) -> str:
    device = next(model.parameters()).device
    idx= torch.tensor([tokenizer.encode(prompt)]).to(device)
    out = generate(model, idx, **kwargs)
    return tokenizer.decode(out[0].tolist())