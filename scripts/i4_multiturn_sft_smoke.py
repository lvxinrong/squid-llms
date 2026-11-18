import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.checkpoints.store import load_checkpoint, save_checkpoint
from squid.data.chat_sft import (
    QwenMultiTurnDataset,
    qwen_multiturn_collate,
)
from squid.device import get_device
from squid.model.v4_qwen3.gpt import (
    GPTModelV4,
    MODEL_V4_DEFAULT_CONFIG,
)
from squid.train.common import calc_loss_loader
from squid.train.sft import sft_train_model

def build_loader(path, tokenizer, batch_size, shuffle):
    data = json.loads(path.read_text(encoding="utf-8"))
    dataset = QwenMultiTurnDataset(data, tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=partial(
            qwen_multiturn_collate,
            pad_token_id=tokenizer.eos_token_id,
        ),
    )

def main():
    torch.manual_seed(123)

    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    context_size = 1024
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    train_loader = build_loader(
        root / "data/instruct/i4_multiturn_smoke_train.json",
        tokenizer,
        batch_size = 2,
        shuffle = True,
    )

    val_loader = build_loader(
        root / "data/instruct/i4_multiturn_smoke_val.json",
        tokenizer,
        batch_size=2,
        shuffle=False,
    )

    model = load_qwen3_dir(
        model_dir,
        context_size = context_size,
        device = device,
    )

    initial_train_loss = calc_loss_loader(
        train_loader, model, device, num_batches = 1
    )

    initial_val_loss = calc_loss_loader(
        val_loader, model, device, num_batches = 1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 1e-5 ,
        weight_decay = 0.0,
    )

    train_losses, val_losses, tokens_seen, learning_rates = (
        sft_train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            n_epochs=2,
            eval_freq=1,
            eval_iter=1,
            warmup_steps=1,
            initial_lr=1e-6,
            min_lr=1e-7,
        )
    )

    model_cfg = {
        **MODEL_V4_DEFAULT_CONFIG,
        "context_length": context_size,
    }

    checkpoint_path = (
            root / "data/model_pth/qwen3/i4-multiturn-sft-smoke.pth"
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        checkpoint_path,
        model,
        model_cfg,
        train_samples=4,
        val_samples=2,
        n_epochs=2,
        initial_train_loss=initial_train_loss,
        initial_val_loss=initial_val_loss,
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
    )

    checkpoint = load_checkpoint(checkpoint_path)
    restored_model = GPTModelV4(checkpoint["config"])
    restored_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"device: {device}")
    print(f"initial train loss: {initial_train_loss:.6f}")
    print(f"initial val loss: {initial_val_loss:.6f}")
    print(f"final train loss: {train_losses[-1]:.6f}")
    print(f"final val loss: {val_losses[-1]:.6f}")
    print(f"optimization steps: {len(learning_rates)}")
    print(f"checkpoint reload: passed")
    print(f"checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()

