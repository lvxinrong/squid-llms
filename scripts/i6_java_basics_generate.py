import argparse
import gc
import hashlib
import json
import runpy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device
from squid.generate.chat import build_qwen_chat_prompt
from squid.generate.engine import generate


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SIZE = 4096
MAX_NEW_TOKENS = 256


def task_prompt(tokenizer, task):
    task_text = f"""你是一名严谨的 Java 代码生成助手。请根据方法签名和规格实现一个短方法。

只输出一个完整 Java 方法，方法签名必须与输入一致。
不要输出类、测试、Markdown 或解释。

[方法签名]
{task["signature"]}

[任务规格]
{task["specification"]}

[公开示例]
{task["public_example"]}"""
    return build_qwen_chat_prompt(
        tokenizer,
        [],
        task_text,
        enable_thinking=False,
    )


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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_java_basics_reference.py"
    ))
    tasks = reference_namespace["load_tasks"](args.tasks)
    evaluate_method = reference_namespace["evaluate_method"]

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_qwen3_dir(
        args.model_dir,
        context_size=CONTEXT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval()
    dtype = str(next(model.parameters()).dtype)

    results = []

    for task in tasks:
        prompt = task_prompt(tokenizer, task)
        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
        )
        prompt_tokens = input_ids.shape[1]
        assert prompt_tokens + MAX_NEW_TOKENS <= CONTEXT_SIZE
        ids_hash = input_hash(input_ids)
        input_ids = input_ids.to(device)

        generated = generate(
            model,
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            context_size=CONTEXT_SIZE,
            temperature=0.0,
            eos_id=tokenizer.eos_token_id,
        )
        generated_ids = generated[0, input_ids.shape[1]:].tolist()
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

        result = evaluate_method(
            task,
            response,
            "original",
        )
        result.update({
            "model_dir": args.model_dir,
            "adapter": None,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "input_ids_sha256": ids_hash,
            "generated_ids": generated_ids,
            "generated_tokens": len(generated_ids),
            "finish_reason": "stop" if stopped else "length",
            "response": response,
            "dtype": dtype,
        })

        if not stopped:
            result["status"] = "length_limited"
            result["behavior_pass"] = False

        report_path = output_dir / f"{task['id'].lower()}-original.json"
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            task["id"],
            result["status"],
            result["finish_reason"],
            f"report: {report_path}",
        )
        results.append(result)

    passed = sum(
        result["status"] == "passed"
        for result in results
    )
    print(f"original model: {passed}/{len(results)} behavior passed")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
