import json
import re
from pathlib import Path

from tqdm import tqdm

from squid.data.instruct import format_input
from squid.generate.text import generate_text


def parse_integer_response(text):
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def evaluate_arithmetic(
    model,
    tokenizer,
    test_data,
    output_path,
    context_size=1024,
    max_new_tokens=16,
    description="Arithmetic evaluation",
):
    items = []

    for entry in tqdm(test_data, desc=description):
        prompt = format_input({
            "instruction": entry["question"],
            "input": "",
        }) + "\n\n### Response:\n"
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=0.0,
            eos_id=tokenizer.eos_id,
        )
        response = generated[len(prompt):].replace("<|endoftext|>", "").strip()
        prediction = parse_integer_response(response)
        items.append({
            **entry,
            "model_response": response,
            "prediction": prediction,
            "correct": prediction == entry["answer"],
        })

    operator_summary = {}
    for operator in ["+", "-", "*", "/"]:
        operator_items = [item for item in items if item["operator"] == operator]
        correct = sum(item["correct"] for item in operator_items)
        numeric = sum(item["prediction"] is not None for item in operator_items)
        operator_summary[operator] = {
            "total": len(operator_items),
            "correct": correct,
            "accuracy": correct / len(operator_items),
            "numeric_format_valid": numeric,
            "numeric_format_rate": numeric / len(operator_items),
        }

    correct_count = sum(item["correct"] for item in items)
    numeric_count = sum(item["prediction"] is not None for item in items)
    summary = {
        "total": len(items),
        "correct": correct_count,
        "accuracy": correct_count / len(items),
        "numeric_format_valid": numeric_count,
        "numeric_format_rate": numeric_count / len(items),
        "by_operator": operator_summary,
    }
    report = {"summary": summary, "items": items}

    Path(output_path).write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return report
