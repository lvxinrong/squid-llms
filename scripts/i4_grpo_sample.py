import argparse
import json
import runpy
from pathlib import Path

import torch

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device
from squid.generate.chat import build_qwen_chat_prompt
from squid.generate.engine import generate_stream
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter
from squid.checkpoints.store import load_checkpoint
from squid.model.v4_qwen3.gpt import GPTModelV4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--checkpoint", default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    device = get_device()

    cot_namespace = runpy.run_path(
        str(root / "scripts/i4_cot_behavior_eval.py")
    )
    baseline_namespace = runpy.run_path(
        str(root / "scripts/i4_behavior_baseline.py")
    )
    split_cot_response = cot_namespace["split_cot_response"]
    has_complete_cot_format = cot_namespace["has_complete_cot_format"]
    evaluate_case = baseline_namespace["evaluate_case"]

    if args.checkpoint:
        checkpoint_path = (
            root / "data/model_pth/qwen3" / args.checkpoint
        )
        checkpoint = load_checkpoint(checkpoint_path)
        model_cfg = {
            **checkpoint["config"],
            "context_length": 1024,
        }
        model = GPTModelV4(model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    else:
        model = load_qwen3_dir(
            model_dir,
            context_size=1024,
            device=device,
        )
    tokenizer = QwenTokenizerAdapter(str(model_dir))

    user_input = "37 × 14 等于多少？请简短推理并给出最终数字。"
    expected_answer = "518"

    prompt = build_qwen_chat_prompt(
        tokenizer.tokenizer,
        [],
        user_input,
        enable_thinking=True,
    )
    prompt_ids = tokenizer.encode(prompt)

    responses = []

    for index in range(args.group_size):
        idx = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )
        stream = generate_stream(
            model,
            idx,
            max_new_tokens=args.max_new_tokens,
            context_size=1024,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=tokenizer.eos_id,
        )

        generated_ids = [
            next_id.item() for next_id in stream
        ]
        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        )
        for marker in ("<|im_end|>", "<|endoftext|>"):
            response = response.split(marker, 1)[0]
        response = response.strip()

        _, final_answer = split_cot_response(response)
        answer_correct, numeric_format, _, _ = evaluate_case(
            {"expected_exact": expected_answer},
            final_answer,
        )
        structure_complete = has_complete_cot_format(response)

        if not structure_complete:
            reward = 0.0
        elif answer_correct and numeric_format:
            reward = 1.0
        elif answer_correct:
            reward = 0.7
        else:
            reward = 0.1

        responses.append({
            "index": index,
            "text": response,
            "final_answer": final_answer,
            "reward": reward,
            "answer_correct": answer_correct,
            "numeric_format": numeric_format,
            "structure_complete": structure_complete,
        })

        print(
            f"\nresponse {index}: "
            f"answer={final_answer}, "
            f"correct={answer_correct}, "
            f"structure={structure_complete}, "
            f"reward={reward}"
        )
        print(response)

    rewards = torch.tensor(
        [item["reward"] for item in responses],
        dtype=torch.float32,
    )
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    advantages = (rewards - mean) / (std + 1e-8)

    for item, advantage in zip(responses, advantages):
        item["advantage"] = advantage.item()

    report = {
        "config": {
            "seed": args.seed,
            "group_size": args.group_size,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        },
        "user_input": user_input,
        "expected_answer": expected_answer,
        "responses": responses,
        "mean_reward": mean.item(),
        "std_reward": std.item(),
        "has_learning_signal": bool(std > 1e-8),
    }

    report_path = root / "data/reports/i4-grpo-sampled-group.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nmean reward: {mean.item():.4f}")
    print(f"std reward: {std.item():.4f}")
    print(f"advantages: {[item['advantage'] for item in responses]}")
    print(f"has learning signal: {bool(std > 1e-8)}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
