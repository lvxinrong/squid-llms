import math
from pathlib import Path
import torch
from squid.data.pretrain import create_pretrain_loader
from squid.device import get_device
from squid.model.gpt import GPTModel
from squid.tokenizer.my_bpe_tokenizer import MyBPETokenizer
from squid.train.common import calc_loss_loader
from squid.train.pretrain import pretrain_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "blog"

def train_model_with_nums_merge(num_merges: str):
    vocab_data = ROOT / "data" / "blog" / num_merges
    tokenizer = MyBPETokenizer.from_directory(vocab_data)
    cfg = {
        "vocab_size": len(tokenizer.vocab),  # 该组词表的实际大小
        "context_length": 128,
        "emb_dim": 64,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    device = get_device()
    model = GPTModel(cfg)
    model.to(device)

    train_text = open(DATA / "train.txt", encoding="utf-8").read()
    val_text = open(DATA / "heldout-手搓LLM笔记07-指令微调.txt", encoding="utf-8").read()
    val_text = val_text + "\n" + open(DATA / "heldout-我想把家里的设备组成一个私人AI实验室.txt",
                                      encoding="utf-8").read()

    train_loader = create_pretrain_loader(train_text, tokenizer, 128, 128,
                                          batch_size=8, shuffle=True)
    val_loader = create_pretrain_loader(val_text, tokenizer, 128, 128,
                                        batch_size=8, shuffle=False, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters())
    num_epochs = 100
    train_losses, val_losses, tokens_seen = pretrain_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, device=device, num_epochs=num_epochs, max_steps=500, eval_freq=100, eval_iter=100
    )
    val_loss = calc_loss_loader(val_loader, model, device)
    density = len(val_text.encode("utf-8")) / len(tokenizer.encode(val_text))
    bpb = val_loss / math.log(2) / density
    print(f"val_loss={val_loss:.3f}  bytes/token={density:.2f}  BPB={bpb:.3f}")


if __name__ == '__main__':
    train_model_with_nums_merge("m512")
    train_model_with_nums_merge("m2000")
    train_model_with_nums_merge("m4000")
    train_model_with_nums_merge("m8000")