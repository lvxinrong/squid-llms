import json
import random
import runpy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.checkpoints.store import load_checkpoint, save_checkpoint
from squid.device import get_device
from squid.model.v4_qwen3.gpt import (
    GPTModelV4,
    MODEL_V4_DEFAULT_CONFIG,
)

# 使用动态脚本不通过import方式调用 i4_dpo_logprob_probe中的 sequence_logprob encode_response 两个函数
def prepare_pairs(path, tokenizer, reference_model, device):
    #  Python 中一种动态执行脚本文件并捕获其命名空间的常见写法
    #  runpy.run_path 函数，它能够在不触发传统 import 机制的情况下运行一个 Python 文件，并返回该文件执行后产生的所有全局变量和定义。
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parent / "i4_dpo_logprob_probe.py")
    )
    encode_response = namespace["encode_response"]
    sequence_logprob = namespace["sequence_logprob"]

    data = json.loads(path.read_text(encoding="utf-8"))
    prepared = []

    for entry in data:
        item = {
            "pair_type": entry["pair_type"],
        }

        for key in ("chosen", "rejected"):
            input_ids, target_ids = encode_response(
                tokenizer,
                entry["messages"],
                entry[key],
            )
            item[f"{key}_input_ids"] = input_ids
            item[f"{key}_target_ids"] = target_ids

            reference_logprob, _ = sequence_logprob(
                reference_model,
                input_ids,
                target_ids,
                device,
            )
            item[f"reference_{key}"] = reference_logprob.item()

        prepared.append(item)

    return prepared


def compute_batch(policy_model, batch, device, beta, require_grad):
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parent / "i4_dpo_logprob_probe.py")
    )
    sequence_logprob = namespace["sequence_logprob"]

    losses = []
    scores = []
    margins = []

    for item in batch:
        policy_logprobs = {}

        for key in ("chosen", "rejected"):
            policy_logprobs[key], _ = sequence_logprob(
                policy_model,
                item[f"{key}_input_ids"],
                item[f"{key}_target_ids"],
                device,
                require_grad=require_grad,
            )

        reference_chosen = torch.tensor(
            item["reference_chosen"],
            device=device,
        )
        reference_rejected = torch.tensor(
            item["reference_rejected"],
            device=device,
        )

        score = (
            (policy_logprobs["chosen"] - reference_chosen)
            - (policy_logprobs["rejected"] - reference_rejected)
        )
        loss = -torch.nn.functional.logsigmoid(beta * score)
        margin = policy_logprobs["chosen"] - policy_logprobs["rejected"]

        losses.append(loss)
        scores.append(score.detach())
        margins.append(margin.detach())

    return (
        torch.stack(losses).mean(),
        torch.stack(scores).mean(),
        torch.stack(margins).mean(),
    )

def evaluate(policy_model, data, device, beta, batch_size):
    policy_model.eval()
    losses = []
    scores = []
    margins = []

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            loss, score, margin = compute_batch(
                policy_model,
                batch,
                device,
                beta,
                require_grad=False,
            )
            losses.append(loss.item())
            scores.append(score.item())
            margins.append(margin.item())

    return {
        "loss": sum(losses) / len(losses),
        "score": sum(scores) / len(scores),
        "margin": sum(margins) / len(margins),
    }


def main():
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)

    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    train_path = root / "data/instruct/i4_dpo_synthetic_train.json"
    val_path = root / "data/instruct/i4_dpo_synthetic_val.json"

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    reference_model = load_qwen3_dir(
        model_dir,
        context_size=1024,
        device=device,
    )

    train_data = prepare_pairs(
        train_path,
        tokenizer,
        reference_model,
        device,
    )
    val_data = prepare_pairs(
        val_path,
        tokenizer,
        reference_model,
        device,
    )

    del reference_model

    policy_model = load_qwen3_dir(
        model_dir,
        context_size=1024,
        device=device,
    )

    batch_size = 2
    max_steps = 16
    eval_freq = 1
    beta = 0.1
    learning_rate = 1e-7

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    initial_val = evaluate(
        policy_model,
        val_data,
        device,
        beta,
        batch_size,
    )

    shuffled = train_data.copy()
    random.shuffle(shuffled)
    batches = [
        shuffled[start:start + batch_size]
        for start in range(0, len(shuffled), batch_size)
    ]

    history = []

    for step, batch in enumerate(batches[:max_steps]):
        policy_model.train()

        loss, score, margin = compute_batch(
            policy_model,
            batch,
            device,
            beta,
            require_grad=True,
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_model.parameters(),
            max_norm=1.0,
            foreach=False,
        )
        optimizer.step()

        val = evaluate(
            policy_model,
            val_data,
            device,
            beta,
            batch_size,
        )

        record = {
            "step": step,
            "train_loss": loss.item(),
            "train_score": score.item(),
            "train_margin": margin.item(),
            "grad_norm_before_clip": grad_norm.item(),
            "val_loss": val["loss"],
            "val_score": val["score"],
            "val_margin": val["margin"],
        }
        history.append(record)

        print(
            f"step={step}, "
            f"train_loss={loss.item():.6f}, "
            f"train_score={score.item():.4f}, "
            f"val_loss={val['loss']:.6f}, "
            f"val_score={val['score']:.4f}, "
            f"grad_norm={grad_norm.item():.2f}"
        )

    model_cfg = {
        **MODEL_V4_DEFAULT_CONFIG,
        "context_length": 1024,
    }
    checkpoint_path = root / "data/model_pth/qwen3/i4-dpo-v1.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        checkpoint_path,
        policy_model,
        model_cfg,
        optimization_steps=max_steps,
        beta=beta,
        learning_rate=learning_rate,
    )

    checkpoint = load_checkpoint(checkpoint_path)
    restored_model = GPTModelV4(checkpoint["config"])
    restored_model.load_state_dict(checkpoint["model_state_dict"])

    report = {
        "config": {
            "model": "Qwen3-0.6B",
            "device": str(device),
            "seed": seed,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "beta": beta,
            "learning_rate": learning_rate,
            "grad_clip": 1.0,
        },
        "initial_val": initial_val,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_reload": "passed",
        "boundary": "仅验证 DPO 小批量训练链路，不代表行为已经改善。",
    }

    report_path = root / "data/reports/i4-dpo-v1-training.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("checkpoint reload: passed")
    print(f"checkpoint: {checkpoint_path}")
    print(f"training report: {report_path}")

if __name__ == '__main__':
    main()