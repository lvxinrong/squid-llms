import json
import runpy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device


def main():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    device = get_device()
    beta = 0.1

    namespace = runpy.run_path(
        str(root / "scripts/i4_dpo_logprob_probe.py")
    )
    encode_response = namespace["encode_response"]
    sequence_logprob = namespace["sequence_logprob"]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    reference_model = load_qwen3_dir(
        model_dir,
        context_size=1024,
        device=device,
    )
    policy_model = load_qwen3_dir(
        model_dir,
        context_size=1024,
        device=device,
    )
    reference_model.eval()
    policy_model.train()

    entry = json.loads(
        (root / "data/instruct/i4_dpo_preference_probe.json")
        .read_text(encoding="utf-8")
    )[0]

    encoded = {
        key: encode_response(
            tokenizer,
            entry["messages"],
            entry[key],
        )
        for key in ("chosen", "rejected")
    }

    reference_logprobs = {}
    policy_logprobs = {}

    for key, (input_ids, target_ids) in encoded.items():
        reference_logprobs[key], _ = sequence_logprob(
            reference_model,
            input_ids,
            target_ids,
            device,
        )
        policy_logprobs[key], _ = sequence_logprob(
            policy_model,
            input_ids,
            target_ids,
            device,
            require_grad=True,
        )

    score = (
        (policy_logprobs["chosen"] - reference_logprobs["chosen"])
        - (policy_logprobs["rejected"] - reference_logprobs["rejected"])
    )
    loss = -torch.nn.functional.logsigmoid(beta * score)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=1e-7,
        weight_decay=0.0,
    )

    print(f"before score: {score.item():.8f}")
    print(f"before loss: {loss.item():.8f}")

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.sqrt(
        sum(
            parameter.grad.norm() ** 2
            for parameter in policy_model.parameters()
            if parameter.grad is not None
        )
    )
    print(f"grad norm: {grad_norm.item():.6f}")

    torch.nn.utils.clip_grad_norm_(
        policy_model.parameters(),
        max_norm=1.0,
        foreach=False,
    )

    after_clip_norm = torch.sqrt(
        sum(
            parameter.grad.norm() ** 2
            for parameter in policy_model.parameters()
            if parameter.grad is not None
        )
    )
    print(f"after clip norm: {after_clip_norm.item():.6f}")
    optimizer.step()
    policy_model.eval()

    updated_logprobs = {}
    for key, (input_ids, target_ids) in encoded.items():
        updated_logprobs[key], _ = sequence_logprob(
            policy_model,
            input_ids,
            target_ids,
            device,
        )

    updated_score = (
        (updated_logprobs["chosen"] - reference_logprobs["chosen"])
        - (updated_logprobs["rejected"] - reference_logprobs["rejected"])
    )
    updated_loss = -torch.nn.functional.logsigmoid(
        beta * updated_score
    )

    print(f"after score: {updated_score.item():.8f}")
    print(f"after loss: {updated_loss.item():.8f}")
    print(f"score change: {(updated_score - score).item():.8f}")
    print(
        "chosen logprob:",
        f"{policy_logprobs['chosen'].item():.6f}",
        "→",
        f"{updated_logprobs['chosen'].item():.6f}",
    )
    print(
        "rejected logprob:",
        f"{policy_logprobs['rejected'].item():.6f}",
        "→",
        f"{updated_logprobs['rejected'].item():.6f}",
    )

    assert torch.isfinite(updated_loss)


if __name__ == "__main__":
    main()
