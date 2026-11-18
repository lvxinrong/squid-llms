import json
from datetime import datetime
from functools import partial
from pathlib import Path

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device
from squid.generate.chat import ChatSession, build_qwen_chat_prompt
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter
import re
import argparse
from squid.checkpoints.store import load_checkpoint
from squid.model.v4_qwen3.gpt import GPTModelV4



def normalize_exact_answer(response):
    return response.strip().rstrip("。")


def extract_numeric_answer(response):
    normalized = normalize_exact_answer(response)
    match = re.fullmatch(
        r"(?:.+?=\s*)?(-?\d+(?:\.\d+)?)",
        normalized,
    )
    return match.group(1) if match else None


def evaluate_case(case, response):
    forbidden_hits = [
        text
        for text in case.get("forbidden_contains", [])
        if text in response
    ]

    if "expected_exact" in case:
        answer_correct = (
            extract_numeric_answer(response)
            == case["expected_exact"]
        )
        format_compliant = re.fullmatch(
            r"-?\d+(?:\.\d+)?",
            normalize_exact_answer(response),
        ) is not None
    else:
        answer_correct = all(
            text in response
            for text in case.get("expected_contains", [])
        )
        format_compliant = True

    passed = answer_correct and format_compliant
    return answer_correct, format_compliant, passed, forbidden_hits

def normalize_copy_text(text):
    return "".join(text.split()).rstrip("。")


def is_previous_assistant_copy(case, response):
    if not case["history"]:
        return False

    previous_assistant = case["history"][-1][1]
    response = normalize_copy_text(response)
    previous_assistant = normalize_copy_text(previous_assistant)

    return bool(response) and response == previous_assistant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--report-name",
        default="i4-qwen3-0_6b-behavior-baseline-v3.json",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    eval_path = root / "data/eval/i4_behavior_eval.json"
    cases = json.loads(eval_path.read_text(encoding="utf-8"))

    context_size = 8192
    max_new_tokens = 512
    device = get_device()

    if args.checkpoint:
        checkpoint_path = (
                root / "data/model_pth/qwen3" / args.checkpoint
        )
        checkpoint = load_checkpoint(checkpoint_path)
        model_cfg = {
            **checkpoint["config"],
            "context_length": context_size,
        }
        model = GPTModelV4(model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        model_label = args.checkpoint
    else:
        model = load_qwen3_dir(
            model_dir,
            context_size=context_size,
            device=device,
        )
        model_label = "Qwen3-0.6B original"

    tokenizer = QwenTokenizerAdapter(str(model_dir))
    prompt_builder = partial(
        build_qwen_chat_prompt,
        tokenizer.tokenizer,
        enable_thinking=False,
    )

    results = []

    for case in cases:
        session = ChatSession(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            prompt_builder=prompt_builder,
        )
        session.history = [
            tuple(turn) for turn in case["history"]
        ]

        prompt = session.prompt_builder(
            session.history,
            case["user"],
        )
        response = session.ask(case["user"])

        (
            answer_correct,
            format_compliant,
            passed,
            forbidden_hits,
        ) = evaluate_case(case, response)

        copied_previous = is_previous_assistant_copy(
            case,
            response,
        )

        results.append({
            "id": case["id"],
            "category": case["category"],
            "history": case["history"],
            "user": case["user"],
            "prompt": prompt,
            "response": response,
            "normalized_response": normalize_exact_answer(response),
            "answer_correct": answer_correct,
            "format_compliant": format_compliant,
            "passed": passed,
            "forbidden_hits": forbidden_hits,
            "empty": not response.strip(),
            "finish_reason": session.last_finish_reason,
            "generated_tokens": session.last_generated_tokens,
            "copied_previous_assistant": copied_previous,
        })

        print(
            f"{case['id']}: "
            f"correct={answer_correct}, "
            f"format={format_compliant}, "
            f"passed={passed}, "
            f"finish={session.last_finish_reason}, "
            f"copy={copied_previous}"
        )

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "config": {
            "model": model_label,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "context_size": context_size,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "enable_thinking": False,
        },
        "cases": results,
        "summary": {
            "total": len(results),
            "answer_correct": sum(
                item["answer_correct"] for item in results
            ),
            "format_compliant": sum(
                item["format_compliant"] for item in results
            ),
            "passed": sum(item["passed"] for item in results),
            "empty": sum(item["empty"] for item in results),
            "natural_stop": sum(
                item["finish_reason"] == "stop"
                for item in results
            ),
            "length_limited": sum(
                item["finish_reason"] == "length"
                for item in results
            ),
            "copied_previous_assistant": sum(
                item["copied_previous_assistant"]
                for item in results
            ),
        },

        "boundary": (
            "仅包含固定行为样例与规则评测，"
            "不代表通用模型能力；contains 仅用于允许自然语言变化的题目。"
        ),
    }

    report_path = root / "data/reports" / args.report_name

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"基线报告已保存到：{report_path}")


if __name__ == "__main__":
    main()
