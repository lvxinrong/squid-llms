import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from squid.checkpoints.gpt2 import load_pretrained_gpt
from squid.checkpoints.store import save_checkpoint, load_model
from squid.data.instruct import instruct_collate, InstructDataset
from squid.device import get_device
from squid.tokenizer import TiktokenTokenizer
from squid.train.common import plot_losses
from squid.train.sft import sft_train_model


def load_pretrained_335m(cfg, path, device):
    return load_pretrained_gpt(cfg, path, device)

def save_335m_instruct_checkpoint(path, model, config, optimizer=None, **meta):
    return save_checkpoint(path, model, config, optimizer, **meta)

def load_335m_instruct_checkpoint(path):
    return load_model(path)

def build_instruct_loader(train_data, test_data, val_data, tokenizer, device, cfg):
    custom_collate_fn = partial(instruct_collate, pad_token_id=tokenizer.eos_id)
    num_workers = 0
    batch_size = 8
    train_loader = DataLoader(
        InstructDataset(train_data, tokenizer),
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        InstructDataset(test_data, tokenizer),
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        InstructDataset(val_data, tokenizer),
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader, val_loader

def plot_training_curve(num_epochs, train_losses, val_losses, tokens_seen):
    if not train_losses:
        return
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

def save_instruct_checkpoint(path, model, cfg):
    return save_checkpoint(path, model, cfg)

def train_instruct_model():
    tokenizer = TiktokenTokenizer("gpt2")
    cfg = {
        "vocab_size": tokenizer.tok.n_vocab,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.05,
        "qkv_bias": True
    }

    train_cfg = {
        "lr": 3e-5,
        "weight_decay": 0.1,
        "n_epochs": 2,
        "eval_freq": 5,
        "eval_iter": 5,
        "initial_lr": 1e-5,
        "min_lr": 1e-6,
    }
    device = get_device()
    pretrained_gpt_path =  Path("/Users/lv/lv_repo/squid-llms/data/model_pth/gpt2/gpt2_355M.pth")
    instruct_checkpoint_path = Path("/Users/lv/lv_repo/squid-llms/data/model_pth/gpt2/gpt2_355M_instruct.pth")
    instruct_train_data_path = Path("/Users/lv/lv_repo/squid-llms/data/instruct/instruction-data.json")
    if not instruct_checkpoint_path.exists():
        # 训练部分
        model = load_pretrained_gpt(cfg, pretrained_gpt_path, device)
        model.to(device)
        torch.manual_seed(123)
        data = json.loads(instruct_train_data_path.read_text(encoding="utf-8"))
        train_portion = int(len(data) * 0.85)
        test_portion = int(len(data) * 0.1)
        train_data = data[: train_portion]
        test_data = data[train_portion : train_portion + test_portion]
        val_data = data[train_portion + test_portion:]
        train_loader, test_loader, val_loader = build_instruct_loader(train_data, test_data, val_data, tokenizer, device, cfg)
        total_steps = len(train_loader) * train_cfg["n_epochs"]
        warmup_steps = int(0.2 * total_steps)

        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
        train_losses, val_losses, track_tokens_seen, track_lrs = sft_train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            n_epochs=train_cfg["n_epochs"],
            eval_freq=train_cfg["eval_freq"],
            eval_iter=train_cfg["eval_iter"],
            warmup_steps=warmup_steps,
            initial_lr=train_cfg["initial_lr"],
            min_lr=train_cfg["min_lr"],
        )

        save_instruct_checkpoint(instruct_checkpoint_path, model, cfg)
        plot_training_curve(train_cfg["n_epochs"], train_losses, val_losses, track_tokens_seen)

if __name__ == "__main__":
    train_instruct_model()
