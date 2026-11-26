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

MINIMAL_TASK_PROMPTS = {
    "J0-01": (
        "请使用 Java 实现一个名为 add 的静态方法，"
        "参数为 int a 和 int b，返回它们的和。"
    ),
    "J0-02": (
        "请使用 Java 实现一个名为 maxOfTwo 的静态方法，"
        "参数为 int a 和 int b，返回较大的那个数；"
        "如果相等，返回任意一个。"
    ),
    "J0-03": (
        "请使用 Java 实现一个名为 sumArray 的静态方法，"
        "参数为 int[] values，返回数组所有元素的和；"
        "空数组返回 0，数组不为 null。"
    ),
    "J0-04": (
        "请使用 Java 实现一个名为 countEven 的静态方法，"
        "参数为 int[] values，返回数组中偶数元素的个数。"
    ),
    "J0-05": (
        "请使用 Java 实现一个名为 toUpperCase 的静态方法，"
        "参数为 String text，返回其大写形式；"
        "输入为 null 时返回 null。"
    ),
    "J0-06": (
        "请使用 Java 实现一个名为 divideAsDouble 的静态方法，"
        "参数为 int dividend 和 int divisor，"
        "返回两数相除的 double 结果，除数不为 0。"
    ),
}


def task_prompt(
    tokenizer,
    task,
    prompt_version="v1",
    generation_mode="instruction",
):
    task_text = f"""你是一名严谨的 Java 代码生成助手。请根据方法签名和规格实现一个短方法。

只输出一个完整 Java 方法，方法签名必须与输入一致。
不要输出类、测试、Markdown 或解释。

[方法签名]
{task["signature"]}

[任务规格]
{task["specification"]}

[公开示例]
{task["public_example"]}"""

    if prompt_version == "v2":
        task_text += """

[完整性要求]
方法体必须非空，并以 } 结束。
不要在方法体完成之前结束生成。"""

    prompt = build_qwen_chat_prompt(
        tokenizer,
        [],
        task_text,
        enable_thinking=False,
    )

    if generation_mode == "raw":
        return (
            "// Task\n"
            f"// {task['specification']}\n"
            "// Public example\n"
            f"// {task['public_example']}\n"
            f"{task['signature']} {{"
        )

    if generation_mode == "minimal":
        return (
            MINIMAL_TASK_PROMPTS[task["id"]]
        )

    if generation_mode == "prefill":
        prompt += task["signature"] + " {"

    return prompt



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
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--prompt-version",
        choices=("v1", "v2"),
        default="v1",
    )
    parser.add_argument(
        "--generation-mode",
        choices=("instruction", "prefill", "raw", "minimal"),
        default="instruction",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_namespace = runpy.run_path(str(
        ROOT / "scripts/i6_java_basics_reference.py"
    ))
    tasks = reference_namespace["load_tasks"](args.tasks)
    evaluate_method = reference_namespace["evaluate_method"]

    device = get_device()
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_qwen3_dir(
        args.model_dir,
        context_size=CONTEXT_SIZE,
        device=device,
        dtype=dtype,
    )
    model.eval()
    dtype = str(next(model.parameters()).dtype)

    results = []

    for task in tasks:
        prompt = task_prompt(
            tokenizer,
            task,
            prompt_version=args.prompt_version,
            generation_mode=args.generation_mode,
        )
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

        completion_response = response
        if args.generation_mode in {"prefill", "raw"}:
            response = (
                task["signature"]
                + " {"
                + completion_response
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
            "generation_mode": args.generation_mode,
            "completion_response": (
                completion_response
                if args.generation_mode in {"prefill", "raw"}
                else None
            ),
            "prompt_format": (
                "minimal_natural_language"
                if args.generation_mode == "minimal"
                else "plain_text"
                if args.generation_mode == "raw"
                else "qwen_chat"
            ),
            "completion_prefix": (
                task["signature"] + " {"
                if args.generation_mode in {"prefill", "raw"}
                else None
            ),
            "evaluator_version": "java-basics-v2-candidate-method",
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
