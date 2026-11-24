import json
import runpy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.lora import load_lora, save_lora
from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device
from squid.train.common import calc_loss_loader
from squid.train.sft import sft_train_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"
DATA_PATH = ROOT / "data/instruct/i6_lang5_smoke.json"
ADAPTER_PATH = (
    ROOT / "data/model_pth/qwen3/i6-lang5-lora-smoke.pth"
)
REPORT_PATH = ROOT / "data/reports/i6-lang5-lora-smoke.json"


def main():
    seed = 123
    torch.manual_seed(seed)

    context_size = 2048
    batch_size = 1
    n_epochs = 1
    peak_lr = 1e-5
    rank = 8
    alpha = 16
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    multiturn_namespace = runpy.run_path(str(
        ROOT / "scripts/i4_multiturn_sft_train.py"
    ))
    build_loader = multiturn_namespace["build_loader"]

    train_loader, train_count = build_loader(
        DATA_PATH,
        tokenizer,
        batch_size,
        shuffle=False,
    )
    diagnostic_loader, diagnostic_count = build_loader(
        DATA_PATH,
        tokenizer,
        batch_size,
        shuffle=False,
    )

    assert train_count == 1
    assert diagnostic_count == 1

    model = load_qwen3_dir(
        MODEL_DIR,
        context_size=context_size,
        device=device,
    )
    base_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    lora_namespace = runpy.run_path(str(
        ROOT / "scripts/i4_lora_sft_train.py"
    ))
    add_lora = lora_namespace["add_lora"]
    add_lora(model, rank=rank, alpha=alpha)

    model.activation_checkpointing = True
    model.to(device)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    initial_train_loss = calc_loss_loader(
        train_loader,
        model,
        device,
        num_batches=1,
    )
    initial_diagnostic_loss = calc_loss_loader(
        diagnostic_loader,
        model,
        device,
        num_batches=1,
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=peak_lr,
        weight_decay=0.0,
    )

    total_steps = len(train_loader) * n_epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    train_losses, diagnostic_losses, tokens_seen, learning_rates = (
        sft_train_model(
            model=model,
            train_loader=train_loader,
            val_loader=diagnostic_loader,
            optimizer=optimizer,
            device=device,
            n_epochs=n_epochs,
            eval_freq=1,
            eval_iter=1,
            warmup_steps=warmup_steps,
            initial_lr=1e-6,
            min_lr=1e-7,
        )
    )

    final_diagnostic_loss = calc_loss_loader(
        diagnostic_loader,
        model,
        device,
        num_batches=1,
    )

    ADAPTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_lora(
        ADAPTER_PATH,
        model,
        rank=rank,
        alpha=alpha,
        train_samples=train_count,
        val_samples=diagnostic_count,
        n_epochs=n_epochs,
        optimization_steps=len(learning_rates),
        final_full_val_loss=final_diagnostic_loss,
    )

    reloaded_model = load_qwen3_dir(
        MODEL_DIR,
        context_size=context_size,
        device=device,
    )
    add_lora(reloaded_model, rank=rank, alpha=alpha)
    load_lora(reloaded_model, ADAPTER_PATH, strict=True)
    reloaded_model.to(device).eval()

    report = {
        "config": {
            "model": "Qwen3-0.6B original",
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
            "activation_checkpointing": True,
        },
        "parameters": {
            "base": base_parameters,
            "trainable": trainable_count,
            "trainable_percent_of_base": (
                trainable_count / base_parameters * 100
            ),
        },
        "initial_train_loss": initial_train_loss,
        "initial_diagnostic_loss": initial_diagnostic_loss,
        "final_train_loss": train_losses[-1],
        "final_diagnostic_loss": final_diagnostic_loss,
        "train_losses": train_losses,
        "diagnostic_losses": diagnostic_losses,
        "tokens_seen": tokens_seen,
        "learning_rates": learning_rates,
        "optimization_steps": len(learning_rates),
        "adapter": str(ADAPTER_PATH),
        "adapter_bytes": ADAPTER_PATH.stat().st_size,
        "adapter_reload": "passed",
        "boundary": (
            "仅一个训练样本、一次 LoRA 更新。"
            "诊断 loss 使用同一训练样本，不是独立验证集；"
            "不用于证明领域修复能力。"
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"trainable parameters: {trainable_count}")
    print(f"trainable percent: {trainable_count / base_parameters * 100:.4f}%")
    print(f"initial train loss: {initial_train_loss:.6f}")
    print(f"final train loss: {train_losses[-1]:.6f}")
    print(f"final diagnostic loss: {final_diagnostic_loss:.6f}")
    print(f"optimization steps: {len(learning_rates)}")
    print(f"adapter bytes: {ADAPTER_PATH.stat().st_size}")
    print("adapter reload: passed")
    print(f"adapter: {ADAPTER_PATH}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
