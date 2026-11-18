import json
import runpy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.lora import load_lora, save_lora
from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device
from squid.model.lora import replace_linear_with_lora
from squid.train.common import calc_loss_loader
from squid.train.sft import sft_train_model

def add_lora(model, rank, alpha):
    for parameter in model.parameters():
        parameter.requires_grad = False

    replace_linear_with_lora(model, rank=rank, alpha=alpha)


def main():
    seed = 123
    torch.manual_seed(seed)

    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    train_path = root / "data/instruct/i4_multiturn_synthetic_train.json"
    val_path = root / "data/instruct/i4_multiturn_synthetic_val.json"

    namespace = runpy.run_path(
        str(root / "scripts/i4_multiturn_sft_train.py")
    )
    build_loader = namespace["build_loader"]

    context_size = 1024
    batch_size = 4
    n_epochs = 1
    peak_lr = 1e-5
    rank = 8
    alpha = 16
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
    base_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    add_lora(model, rank=rank, alpha=alpha)
    model.to(device)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    total_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    initial_train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=4
    )
    initial_val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=4
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
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

    adapter_path = (
        root / "data/model_pth/qwen3/i4-lora-sft-r8a16.pth"
    )
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    save_lora(
        adapter_path,
        model,
        rank=rank,
        alpha=alpha,
        train_samples=train_count,
        val_samples=val_count,
        n_epochs=n_epochs,
        optimization_steps=len(learning_rates),
        final_full_val_loss=final_full_val_loss,
    )

    reloaded_model = load_qwen3_dir(
        model_dir,
        context_size=context_size,
        device=device,
    )
    add_lora(reloaded_model, rank=rank, alpha=alpha)
    load_lora(reloaded_model, adapter_path, strict=True)
    reloaded_model.to(device).eval()

    full_checkpoint_path = (
        root / "data/model_pth/qwen3/i4-multiturn-sft-v2-1epoch.pth"
    )

    report = {
        "config": {
            "model": "Qwen3-0.6B",
            "device": str(device),
            "seed": seed,
            "context_size": context_size,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "peak_lr": peak_lr,
            "initial_lr": 1e-6,
            "min_lr": 1e-7,
            "weight_decay": 0.0,
            "warmup_steps": warmup_steps,
            "rank": rank,
            "alpha": alpha,
        },
        "parameters": {
            "base": base_parameters,
            "trainable": trainable_count,
            "total_with_lora": total_count,
            "trainable_percent_of_base": (
                trainable_count / base_parameters * 100
            ),
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
        "adapter": str(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "full_checkpoint": str(full_checkpoint_path),
        "full_checkpoint_bytes": full_checkpoint_path.stat().st_size,
        "adapter_reload": "passed",
        "boundary": (
            "仅比较当前受控数据上的 LoRA 与全参数训练；"
            "不预先判断 LoRA 一定更优。"
        ),
    }

    report_path = root / "data/reports/i4-lora-sft-r8a16-training.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"trainable parameters: {trainable_count}")
    print(f"trainable percent: {trainable_count / base_parameters * 100:.4f}%")
    print(f"initial train loss: {initial_train_loss:.6f}")
    print(f"initial val loss: {initial_val_loss:.6f}")
    print(f"final train loss: {train_losses[-1]:.6f}")
    print(f"final full val loss: {final_full_val_loss:.6f}")
    print(f"adapter bytes: {adapter_path.stat().st_size}")
    print("adapter reload: passed")
    print(f"adapter: {adapter_path}")
    print(f"training report: {report_path}")


if __name__ == "__main__":
    main()