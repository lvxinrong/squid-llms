import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device
from squid.generate.chat import build_qwen_chat_prompt
from squid.generate.engine import generate, generate_no_cache
import runpy
from squid.checkpoints.lora import load_lora



ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SIZE = 4096
MAX_NEW_TOKENS = 128
REQUIRED_FIELDS = {
    "id",
    "kind",
    "data_role",
    "question",
    "reference_answer",
    "scoring_points",
    "notes",
}


def load_tasks(path):
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(tasks, list) and tasks

    ids = set()
    for task in tasks:
        assert set(task) == REQUIRED_FIELDS, task.get("id")
        assert task["id"] not in ids, task["id"]
        assert task["question"], task["id"]
        assert len(task["scoring_points"]) >= 2, task["id"]
        assert task["data_role"] in {
            "train",
            "validation",
            "eval_holdout",
        }, task["id"]
        ids.add(task["id"])

    return tasks


def input_hash(input_ids):
    raw = (
        input_ids.detach()
        .cpu()
        .contiguous()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--sft-prompt", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.tasks)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model_dtype = getattr(torch, args.dtype)
    model = load_qwen3_dir(
        args.model_dir,
        context_size=CONTEXT_SIZE,
        device=device,
        dtype=model_dtype,
    )

    adapter_rank = None
    adapter_alpha = None

    if args.adapter:
        baseline_namespace = runpy.run_path(str(
            ROOT / "scripts/i6_structured_edit_baseline.py"
        ))
        adapter_rank, adapter_alpha = (
            baseline_namespace["read_adapter_config"](
                args.adapter
            )
        )
        baseline_namespace["add_lora"](
            model,
            rank=adapter_rank,
            alpha=adapter_alpha,
        )
        load_lora(
            model,
            args.adapter,
            strict=True,
        )
        model.to(
            device=device,
            dtype=model_dtype,
        )

    model.eval()
    dtype = str(next(model.parameters()).dtype)

    for task in tasks:
        if args.sft_prompt:
            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是一名严谨的 Java 教师。请直接回答问题，"
                            "先给结论，再用简洁的话解释原因。"
                            "不要重复题目，不要输出与问题无关的内容。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": task["question"],
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            task_text = (
                "你是一名 Java 教师。请直接回答下面的问题，"
                "先给结论，再用简洁的话解释原因。"
                "不要重复题目，不要输出与问题无关的内容。\n\n"
                f"[问题]\n{task['question']}"
            )
            prompt = build_qwen_chat_prompt(
                tokenizer,
                [],
                task_text,
                enable_thinking=False,
            )

        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
        )
        prompt_tokens = input_ids.shape[1]
        assert prompt_tokens + MAX_NEW_TOKENS <= CONTEXT_SIZE
        ids_hash = input_hash(input_ids)
        input_ids = input_ids.to(device)

        if args.no_cache:
            generated = generate_no_cache(
                model,
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                context_size=CONTEXT_SIZE,
                temperature=0.0,
                eos_id=tokenizer.eos_token_id,
            )
        else:
            generated = generate(
                model,
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                context_size=CONTEXT_SIZE,
                temperature=0.0,
                eos_id=tokenizer.eos_token_id,
            )

        generated_ids = generated[0, input_ids.shape[1]:].tolist()
        if args.no_cache:
            stopped = len(generated_ids) < MAX_NEW_TOKENS
            content_ids = generated_ids
        else:
            stopped = bool(
                generated_ids
                and generated_ids[-1] == tokenizer.eos_token_id
            )
            content_ids = (
                generated_ids[:-1]
                if stopped
                else generated_ids
            )

        response = tokenizer.decode(
            content_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        report = {
            "instance_id": task["id"],
            "kind": task["kind"],
            "model_dir": args.model_dir,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "input_ids_sha256": ids_hash,
            "generated_ids": generated_ids,
            "generated_tokens": len(generated_ids),
            "finish_reason": "stop" if stopped else "length",
            "response": response,
            "reference_answer": task["reference_answer"],
            "scoring_points": task["scoring_points"],
            "notes": task["notes"],
            "dtype": dtype,
            "evaluation": {
                "status": "generated" if stopped else "length_limited",
                "manual_score": None,
                "manual_notes": None,
            },
            "adapter": args.adapter,
            "adapter_rank": adapter_rank,
            "adapter_alpha": adapter_alpha,
            "prompt_mode": "sft" if args.sft_prompt else "legacy",
            "generation_mode": "no_cache" if args.no_cache else "kv_cache",

        }

        report_path = output_dir / f"{task['id'].lower()}.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            task["id"],
            report["evaluation"]["status"],
            f"report: {report_path}",
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
