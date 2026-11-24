import argparse
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


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"


def add_lora(model, rank, alpha):
    for parameter in model.parameters():
        parameter.requires_grad = False
    replace_linear_with_lora(
        model,
        rank=rank,
        alpha=alpha,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--best-adapter", default=None)
    args = parser.parse_args()

    seed = 123
    torch.manual_seed(seed)
    device = get_device()

    multiturn_namespace = runpy.run_path(str(
        ROOT / "scripts/i4_multiturn_sft_train.py"
    ))
    build_loader = multiturn_namespace["build_loader"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    train_loader, train_count = build_loader(
        Path(args.train),
        tokenizer,
        args.batch_size,
        shuffle=True,
        enable_thinking=False,
    )
    val_loader, val_count = build_loader(
        Path(args.val),
        tokenizer,
        args.batch_size,
        shuffle=False,
        enable_thinking=False,
    )

    model = load_qwen3_dir(
        MODEL_DIR,
        context_size=args.context_size,
        device=device,
    )
    base_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    add_lora(model, rank=args.rank, alpha=args.alpha)
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
        num_batches=None,
    )
    initial_val_loss = calc_loss_loader(
        val_loader,
        model,
        device,
        num_batches=None,
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    best_val_loss = float("inf")
    best_step = None
    best_epoch = None
    on_evaluate = None

    if args.best_adapter:
        best_adapter_path = Path(args.best_adapter)
        best_adapter_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        def on_evaluate(metrics):
            nonlocal best_val_loss, best_step, best_epoch

            if metrics["val_loss"] >= best_val_loss:
                return

            best_val_loss = metrics["val_loss"]
            best_step = metrics["global_step"] + 1
            best_epoch = metrics["epoch"]

            save_lora(
                best_adapter_path,
                model,
                rank=args.rank,
                alpha=args.alpha,
                train_samples=train_count,
                val_samples=val_count,
                n_epochs=args.epochs,
                optimization_steps=best_step,
                final_full_val_loss=best_val_loss,
            )


    train_losses, val_losses, tokens_seen, learning_rates = (
        sft_train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            n_epochs=args.epochs,
            eval_freq=4,
            eval_iter=1,
            warmup_steps=warmup_steps,
            initial_lr=1e-6,
            min_lr=1e-7,
            on_evaluate=on_evaluate,
        )
    )

    final_train_loss = calc_loss_loader(
        train_loader,
        model,
        device,
        num_batches=None,
    )
    final_val_loss = calc_loss_loader(
        val_loader,
        model,
        device,
        num_batches=None,
    )

    adapter_path = Path(args.adapter)
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    save_lora(
        adapter_path,
        model,
        rank=args.rank,
        alpha=args.alpha,
        train_samples=train_count,
        val_samples=val_count,
        n_epochs=args.epochs,
        optimization_steps=len(learning_rates),
        final_full_val_loss=final_val_loss,
    )

    reloaded_model = load_qwen3_dir(
        MODEL_DIR,
        context_size=args.context_size,
        device=device,
    )
    add_lora(reloaded_model, rank=args.rank, alpha=args.alpha)
    load_lora(reloaded_model, adapter_path, strict=True)
    reloaded_model.to(device).eval()

    report = {
        "config": {
            "model": "Qwen3-0.6B original",
            "device": str(device),
            "seed": seed,
            "context_size": args.context_size,
            "batch_size": args.batch_size,
            "n_epochs": args.epochs,
            "peak_lr": args.learning_rate,
            "initial_lr": 1e-6,
            "min_lr": 1e-7,
            "weight_decay": 0.0,
            "warmup_steps": warmup_steps,
            "rank": args.rank,
            "alpha": args.alpha,
            "activation_checkpointing": True,
            "enable_thinking": False,
        },
        "data": {
            "train_path": args.train,
            "val_path": args.val,
            "train_samples": train_count,
            "val_samples": val_count,
        },
        "parameters": {
            "base": base_count,
            "trainable": trainable_count,
            "trainable_percent_of_base": (
                trainable_count / base_count * 100
            ),
        },
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen,
        "learning_rates": learning_rates,
        "optimization_steps": len(learning_rates),
        "adapter": str(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "adapter_reload": "passed",
        "boundary": (
            f"当前仅为 {train_count} train + {val_count} val "
            "的小批量 LoRA 对照；"
            "不证明 held-out 或未见项目修复能力提升。"
        ),
        "best_adapter": args.best_adapter,
        "best_val_loss": (
            best_val_loss if best_step is not None else None
        ),
        "best_step": best_step,
        "best_epoch": best_epoch,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.best_adapter and best_step is not None:
        best_model = load_qwen3_dir(
            MODEL_DIR,
            context_size=args.context_size,
            device=device,
        )
        add_lora(best_model, rank=args.rank, alpha=args.alpha)
        load_lora(best_model, args.best_adapter, strict=True)
        best_model.to(device).eval()

        print(f"best val loss: {best_val_loss:.6f}")
        print(f"best step: {best_step}")
        print("best adapter reload: passed")


    print(f"trainable parameters: {trainable_count}")
    print(f"initial train loss: {initial_train_loss:.6f}")
    print(f"initial val loss: {initial_val_loss:.6f}")
    print(f"final train loss: {final_train_loss:.6f}")
    print(f"final val loss: {final_val_loss:.6f}")
    print(f"optimization steps: {len(learning_rates)}")
    print(f"adapter bytes: {adapter_path.stat().st_size}")
    print("adapter reload: passed")
    print(f"adapter: {adapter_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
