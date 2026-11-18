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
import torch

from squid.checkpoints.lora import load_lora
from squid.model.lora import replace_linear_with_lora



def split_for_generation(messages):
    history = []

    for index in range(1, len(messages) - 2, 2):
        history.append((
            messages[index]["content"],
            messages[index + 1]["content"],
        ))

    return history, messages[-2]["content"], messages[-1]["content"]


def memory_answer_core(text):
    text = text.strip().rstrip("。")
    prefixes = (
        "你的名字是",
        "你用的名字是",
        "你的首选爱好是",
        "你最喜欢",
        "你偏爱",
        "你最爱",
        "你叫",
        "你喜欢",
    )

    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):].strip()

    return text


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
        "copied_previous_assistant": sum(
            item["copied_previous_assistant"]
            for item in results
        ),
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
        default="data/instruct/i4_multiturn_synthetic_val.json",
    )
    parser.add_argument("--lora-adapter", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    baseline_namespace = runpy.run_path(
        str(root / "scripts/i4_behavior_baseline.py")
    )
    audit_namespace = runpy.run_path(
        str(root / "scripts/i4_synthetic_data_audit.py")
    )

    evaluate_case = baseline_namespace["evaluate_case"]
    is_copy = baseline_namespace["is_previous_assistant_copy"]
    infer_category = audit_namespace["infer_category"]

    context_size = 1024
    max_new_tokens = 128
    device = get_device()
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"

    if args.lora_adapter:
        adapter_path = (
            root / "data/model_pth/qwen3" / args.lora_adapter
        )
        payload = torch.load(
            adapter_path,
            map_location="cpu",
            weights_only=True,
        )

        model = load_qwen3_dir(
            model_dir,
            context_size=context_size,
            device=device,
        )
        for parameter in model.parameters():
            parameter.requires_grad = False

        replace_linear_with_lora(
            model,
            rank=payload["rank"],
            alpha=payload["alpha"],
        )
        load_lora(model, adapter_path, strict=True)
        model.to(device).eval()
        model_label = args.lora_adapter

    elif args.checkpoint:
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

    data_path = root / args.data_path
    data = json.loads(data_path.read_text(encoding="utf-8"))

    results = []

    for index, entry in enumerate(data):
        messages = entry["messages"]
        category = infer_category(messages)
        history, current_user, expected = split_for_generation(messages)

        if category in {"single_arithmetic", "numeric_update"}:
            case = {
                "expected_exact": expected,
                "history": history,
            }
        else:
            case = {
                "expected_contains": [memory_answer_core(expected)],
                "history": history,
            }

        session = ChatSession(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            prompt_builder=prompt_builder,
        )
        original_history = history.copy()
        session.history = history.copy()
        response = session.ask(current_user)
        assert case["history"] == original_history
        (
            answer_correct,
            format_compliant,
            passed,
            forbidden_hits,
        ) = evaluate_case(case, response)
        copied = is_copy(case, response)

        results.append({
            "index": index,
            "category": category,
            "history": history,
            "user": current_user,
            "expected": expected,
            "response": response,
            "answer_correct": answer_correct,
            "format_compliant": format_compliant,
            "passed": passed,
            "forbidden_hits": forbidden_hits,
            "copied_previous_assistant": copied,
            "finish_reason": session.last_finish_reason,
            "generated_tokens": session.last_generated_tokens,
        })

        print(
            f"{index:02d} {category}: "
            f"correct={answer_correct}, "
            f"format={format_compliant}, "
            f"copy={copied}, "
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
            "enable_thinking": False,
        },
        "cases": results,
        "summary": summary,
        "boundary": (
            "仅评估 32 条独立合成验证样本；"
            "数值题使用精确规则，名字和偏好题使用核心实体包含规则，"
            "不代表开放域语义理解与通用能力。"
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
