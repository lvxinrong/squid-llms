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


def build_loader(
    path,
    tokenizer,
    batch_size,
    shuffle,
    enable_thinking=None,
):
    data = json.loads(path.read_text(encoding="utf-8"))
    dataset = QwenMultiTurnDataset(
        data,
        tokenizer,
        enable_thinking=enable_thinking,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=partial(
            qwen_multiturn_collate,
            pad_token_id=tokenizer.eos_token_id,
        ),
    )
    return loader, len(data)

def main():
    seed = 123
    torch.manual_seed(seed)

    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    train_path = root / "data/instruct/i4_multiturn_synthetic_train.json"
    val_path = root / "data/instruct/i4_multiturn_synthetic_val.json"

    context_size = 1024
    batch_size = 4
    n_epochs = 1
    peak_lr = 1e-5
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    train_loader, train_count = build_loader(
        train_path,
        tokenizer,
        batch_size,
        shuffle=True,
    )
    val_loader, val_count = build_loader(
        val_path,
        tokenizer,
        batch_size,
        shuffle=False,
    )

    model = load_qwen3_dir(
        model_dir,
        context_size=context_size,
        device=device,
    )

    initial_train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=4
    )

    initial_val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=4
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        weight_decay=0.0,
    )

    total_steps = len(train_loader) * n_epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    train_losses, val_losses, tokens_seen, learning_rates = (
        sft_train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            n_epochs=n_epochs,
            eval_freq=8,
            eval_iter=4,
            warmup_steps=warmup_steps,
            initial_lr=1e-6,
            min_lr=1e-7,
        )
    )

    final_full_val_loss = calc_loss_loader(
        val_loader,
        model,
        device,
        num_batches=None,
    )

    model_cfg = {
        **MODEL_V4_DEFAULT_CONFIG,
        "context_length": context_size,
    }

    checkpoint_path = (
        root / "data/model_pth/qwen3/i4-multiturn-sft-v2-1epoch.pth"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        checkpoint_path,
        model,
        model_cfg,
        train_samples=train_count,
        val_samples=val_count,
        n_epochs=n_epochs,
        optimization_steps=len(learning_rates),
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
        final_full_val_loss=final_full_val_loss,
    )

    checkpoint = load_checkpoint(checkpoint_path)
    restored_model = GPTModelV4(checkpoint["config"])
    restored_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    report = {
        "config": {
            "model": "Qwen3-0.6B",
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "seed": seed,
            "context_size": context_size,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "peak_lr": peak_lr,
            "initial_lr": 1e-6,
            "min_lr": 1e-7,
            "weight_decay": 0.0,
            "warmup_steps": warmup_steps,
            "eval_freq": 8,
            "eval_iter": 4,
        },
        "data": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_samples": train_count,
            "val_samples": val_count,
        },
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_full_val_loss": final_full_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen,
        "learning_rates": learning_rates,
        "optimization_steps": len(learning_rates),
        "checkpoint": str(checkpoint_path),
        "checkpoint_reload": "passed",
        "boundary": (
            "仅用于受控多轮合成数据 SFT；"
            "loss 结果不代表行为或通用能力已经提升。"
        ),
    }

    report_path = (
        root / "data/reports/i4-multiturn-sft-v2-1epoch-training.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"initial train loss: {initial_train_loss:.6f}")
    print(f"initial val loss: {initial_val_loss:.6f}")
    print(f"final train loss: {train_losses[-1]:.6f}")
    print(f"final full val loss: {final_full_val_loss:.6f}")
    print(f"optimization steps: {len(learning_rates)}")
    print("checkpoint reload: passed")
    print(f"checkpoint: {checkpoint_path}")
    print(f"training report: {report_path}")

if __name__ == '__main__':
    main()
