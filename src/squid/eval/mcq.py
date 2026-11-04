import json
from pathlib import Path

import torch

@torch.no_grad()
def predict_choice(model, tokenizer, prompt, labels=("A", "B", "C", "D"), ):
    device = next(model.parameters()).device
    idx = tokenizer.encode(prompt)
    idx = torch.tensor(idx).unsqueeze(0).to(device)

    logits = model(idx)
    next_logits = logits[0, -1, :]  # (vocab_size,)

    candidate_ids = []
    for label in labels:
        token_ids = tokenizer.encode(" " + label)
        if len(token_ids) != 1:
            raise ValueError(f"{label!r} 必须恰好编码为一个 token")
        candidate_ids.append(token_ids[0])
    candidate_scores = next_logits[candidate_ids]# (候选数量,)
    pred_index = candidate_scores.argmax().item()
    return pred_index


def format_mc_prompt(question: str, options: list[str]) -> str:
    lines = [f"Question: {question}"]
    for i, opt in enumerate(options):
        label = chr(ord('A') + i)   # A, B, C, ...
        lines.append(f"{label}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


def make_labels(count: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(count)]

def evaluate_mcq(model, tokenizer, test_data, output_path):
    correct_count = 0

    for index, entry in enumerate(test_data):
        question = entry["question"]
        labels = [
            chr(ord("A") + i)
            for i in range(len(entry["options"]))
        ]
        prompt = format_mc_prompt(question, entry["options"])
        prediction = predict_choice(model, tokenizer, prompt, labels)

        entry["prediction"] = prediction
        is_correct = prediction == entry["answer"]
        entry["correct"] = is_correct

        if is_correct:
            correct_count += 1

    total = len(test_data)
    accuracy = correct_count / total if total else 0.0
    report = {
        "summary": {
            "total": total,
            "correct": correct_count,
            "accuracy": accuracy,
        },
        "items": test_data,
    }

    Path(output_path).write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


@torch.no_grad()
def score_continuation(model, tokenizer, prompt, continuation):
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt)
    continuation_ids = tokenizer.encode(" " + continuation)

    if not continuation_ids:
        raise ValueError("候选答案不能为空")

    input_ids = torch.tensor(
        [prompt_ids + continuation_ids],
        dtype=torch.long,
        device=device,
    )
    logits = model(input_ids)

    start = len(prompt_ids) - 1
    answer_logits = logits[:, start:start + len(continuation_ids), :]
    targets = torch.tensor(continuation_ids, dtype=torch.long, device=device)

    log_probs = torch.log_softmax(answer_logits[0], dim=-1)
    token_log_probs = log_probs.gather(
        1,
        targets.unsqueeze(1),
    ).squeeze(1)

    sum_logprob = token_log_probs.sum().item()
    mean_logprob = token_log_probs.mean().item()
    return sum_logprob, mean_logprob


def evaluate_answer_likelihood(model, tokenizer, test_data, output_path):
    sum_correct = 0
    mean_correct = 0

    for entry in test_data:
        prompt = f"Question: {entry['question']}\nAnswer:"
        candidates = []

        for option in entry["options"]:
            sum_logprob, mean_logprob = score_continuation(
                model,
                tokenizer,
                prompt,
                option,
            )
            candidates.append({
                "option": option,
                "sum_logprob": sum_logprob,
                "mean_logprob": mean_logprob,
            })

        sum_prediction = max(
            range(len(candidates)),
            key=lambda index: candidates[index]["sum_logprob"],
        )
        mean_prediction = max(
            range(len(candidates)),
            key=lambda index: candidates[index]["mean_logprob"],
        )

        entry["candidates"] = candidates
        entry["sum_prediction"] = sum_prediction
        entry["mean_prediction"] = mean_prediction
        entry["sum_correct"] = sum_prediction == entry["answer"]
        entry["mean_correct"] = mean_prediction == entry["answer"]

        if entry["sum_correct"]:
            sum_correct += 1
        if entry["mean_correct"]:
            mean_correct += 1

    total = len(test_data)
    report = {
        "summary": {
            "total": total,
            "sum_correct": sum_correct,
            "sum_accuracy": sum_correct / total if total else 0.0,
            "mean_correct": mean_correct,
            "mean_accuracy": mean_correct / total if total else 0.0,
        },
        "items": test_data,
    }

    Path(output_path).write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return report
