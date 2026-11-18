import json
import runpy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.store import load_checkpoint, save_checkpoint
from squid.device import get_device
from squid.model.v4_qwen3.gpt import (
    GPTModelV4,
    MODEL_V4_DEFAULT_CONFIG,
)


def grpo_loss(
    policy_model,
    encoded,
    old_logprobs,
    advantages,
    device,
    epsilon,
    sequence_logprob,
):
    current_logprobs = []

    for input_ids, target_ids in encoded:
        logprob, _ = sequence_logprob(
            policy_model,
            input_ids,
            target_ids,
            device,
            require_grad=True,
        )
        current_logprobs.append(logprob)

    current_logprobs = torch.stack(current_logprobs)
    ratio = torch.exp(current_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - epsilon,
        1.0 + epsilon,
    )

    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    loss = -torch.min(unclipped, clipped).mean()

    return loss, current_logprobs, ratio


def main():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    checkpoint_path = (
        root / "data/model_pth/qwen3/i4-cot-sft-v2-lr1e-6.pth"
    )
    group_path = root / "data/reports/i4-grpo-sampled-group.json"

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    namespace = runpy.run_path(
        str(root / "scripts/i4_dpo_logprob_probe.py")
    )
    encode_response = namespace["encode_response"]
    sequence_logprob = namespace["sequence_logprob"]

    checkpoint = load_checkpoint(checkpoint_path)
    model_cfg = {
        **checkpoint["config"],
        "context_length": 1024,
    }
    policy_model = GPTModelV4(model_cfg)
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    policy_model.to(device)
    policy_model.eval()

    group = json.loads(group_path.read_text(encoding="utf-8"))
    assert group["has_learning_signal"]

    messages = [
        {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
        {"role": "user", "content": group["user_input"]},
    ]

    encoded = []
    old_logprobs = []

    for response in group["responses"]:
        input_ids, target_ids = encode_response(
            tokenizer,
            messages,
            response["text"],
        )
        encoded.append((input_ids, target_ids))

        logprob, _ = sequence_logprob(
            policy_model,
            input_ids,
            target_ids,
            device,
        )
        old_logprobs.append(logprob.detach())

    old_logprobs = torch.stack(old_logprobs)
    advantages = torch.tensor(
        [response["advantage"] for response in group["responses"]],
        dtype=torch.float32,
        device=device,
    )

    epsilon = 0.2
    learning_rate = 1e-7

    policy_model.train()
    before_loss, _, before_ratio = grpo_loss(
        policy_model,
        encoded,
        old_logprobs,
        advantages,
        device,
        epsilon,
        sequence_logprob,
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    optimizer.zero_grad()
    before_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy_model.parameters(),
        max_norm=1.0,
        foreach=False,
    )
    optimizer.step()
    policy_model.eval()

    with torch.no_grad():
        after_loss, after_logprobs, after_ratio = grpo_loss(
            policy_model,
            encoded,
            old_logprobs,
            advantages,
            device,
            epsilon,
            sequence_logprob,
        )

    print(f"before loss: {before_loss.item():.8f}")
    print(f"after loss: {after_loss.item():.8f}")
    print(f"grad norm before clip: {grad_norm.item():.4f}")
    print(f"before mean ratio: {before_ratio.mean().item():.6f}")
    print(f"after mean ratio: {after_ratio.mean().item():.6f}")

    for index, response in enumerate(group["responses"]):
        print(
            f"response {index}: "
            f"advantage={response['advantage']:.2f}, "
            f"old={old_logprobs[index].item():.4f}, "
            f"new={after_logprobs[index].item():.4f}, "
            f"ratio={after_ratio[index].item():.6f}"
        )

    assert torch.isfinite(after_loss)

    output_checkpoint = (
        root / "data/model_pth/qwen3/i4-grpo-one-group.pth"
    )
    save_checkpoint(
        output_checkpoint,
        policy_model,
        {
            **MODEL_V4_DEFAULT_CONFIG,
            "context_length": 1024,
        },
        optimization_steps=1,
        epsilon=epsilon,
        learning_rate=learning_rate,
        reward_version="shaped_v2",
    )

    reloaded = GPTModelV4({
        **MODEL_V4_DEFAULT_CONFIG,
        "context_length": 1024,
    })
    reloaded.load_state_dict(
        load_checkpoint(output_checkpoint)["model_state_dict"]
    )

    report = {
        "config": {
            "base_checkpoint": str(checkpoint_path),
            "group_size": len(encoded),
            "epsilon": epsilon,
            "learning_rate": learning_rate,
            "grad_clip": 1.0,
            "reward_version": "shaped_v2",
            "optimization_steps": 1,
            "reference_kl": False,
        },
        "before_loss": before_loss.item(),
        "after_loss": after_loss.item(),
        "grad_norm_before_clip": grad_norm.item(),
        "old_logprobs": old_logprobs.tolist(),
        "new_logprobs": after_logprobs.tolist(),
        "after_ratios": after_ratio.tolist(),
        "rewards": [item["reward"] for item in group["responses"]],
        "advantages": advantages.tolist(),
        "checkpoint": str(output_checkpoint),
        "checkpoint_reload": "passed",
        "boundary": (
            "仅完成一组、一次、序列级 GRPO 更新；"
            "未加入参考模型 KL，不代表行为已经提升。"
        ),
    }

    report_path = root / "data/reports/i4-grpo-one-update.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("checkpoint reload: passed")
    print(f"checkpoint: {output_checkpoint}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
