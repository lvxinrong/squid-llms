import argparse
import json
import runpy
from functools import partial
from pathlib import Path

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.checkpoints.store import load_checkpoint
from squid.device import get_device
from squid.generate.chat import ChatSession, build_qwen_chat_prompt
from squid.model.v4_qwen3.gpt import GPTModelV4
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter


def split_cot_response(response):
    if "</think>" not in response:
        return "", response.strip()

    reasoning, final_answer = response.rsplit("</think>", 1)
    if "<think>" in reasoning:
        reasoning = reasoning.split("<think>", 1)[1]

    return reasoning.strip(), final_answer.strip()


def has_complete_cot_format(response):
    stripped = response.strip()
    if not stripped.startswith("<think>"):
        return False

    reasoning, final_answer = split_cot_response(stripped)

    return (
        response.count("<think>") == 1
        and response.count("</think>") == 1
        and bool(reasoning)
        and bool(final_answer)
        and reasoning != final_answer
    )


def summarize(results):
    return {
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--report-name", required=True)
    parser.add_argument(
        "--data-path",
        default="data/instruct/i4_cot_synthetic_val.json",
    )
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    evaluate_case = runpy.run_path(
        str(root / "scripts/i4_behavior_baseline.py")
    )["evaluate_case"]
    split_for_generation = runpy.run_path(
        str(root / "scripts/i4_synthetic_behavior_eval.py")
    )["split_for_generation"]

    context_size = 1024
    max_new_tokens = 512
    device = get_device()
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"

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
        enable_thinking=True,
    )

    data_path = root / args.data_path
    data = json.loads(data_path.read_text(encoding="utf-8"))

    if args.limit is not None:
        data = data[:args.limit]

    results = []

    for index, entry in enumerate(data):
        messages = entry["messages"]
        category = (
            "single_arithmetic"
            if len(messages) == 3
            else "numeric_update"
        )
        history, current_user, expected = split_for_generation(messages)

        expected_answer = split_cot_response(expected)[1]
        assert expected_answer.isdigit()

        session = ChatSession(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            prompt_builder=prompt_builder,
        )
        session.history = history.copy()

        response = session.ask(current_user)
        reasoning, final_answer = split_cot_response(response)

        (
            answer_correct,
            numeric_format,
            _,
            forbidden_hits,
        ) = evaluate_case(
            {"expected_exact": expected_answer},
            final_answer,
        )

        format_compliant = (
            has_complete_cot_format(response)
            and numeric_format
        )
        passed = answer_correct and format_compliant

        results.append({
            "index": index,
            "category": category,
            "history": history,
            "user": current_user,
            "expected": expected,
            "response": response,
            "reasoning": reasoning,
            "final_answer": final_answer,
            "answer_correct": answer_correct,
            "format_compliant": format_compliant,
            "passed": passed,
            "forbidden_hits": forbidden_hits,
            "empty": not response.strip(),
            "finish_reason": session.last_finish_reason,
            "generated_tokens": session.last_generated_tokens,
            "expected_answer": expected_answer,
        })

        print(
            f"{index:02d} {category}: "
            f"correct={answer_correct}, "
            f"format={format_compliant}, "
            f"finish={session.last_finish_reason}, "
            f"response={response!r}"
        )

    categories = sorted({item["category"] for item in results})
    summary = summarize(results)
    summary["by_category"] = {
        category: summarize([
            item for item in results
            if item["category"] == category
        ])
        for category in categories
    }

    report = {
        "config": {
            "model": model_label,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "context_size": context_size,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "enable_thinking": True,
        },
        "cases": results,
        "summary": summary,
        "boundary": (
            "仅评估 32 条独立 CoT 验证样本；"
            "答案正确性与 CoT 格式分开统计，"
            "当前不证明推理过程本身的语义质量。"
        ),
    }

    report_path = root / "data/reports" / args.report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False))
    print(f"报告已保存到：{report_path}")


if __name__ == "__main__":
    main()
