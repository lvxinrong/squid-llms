import argparse
import hashlib
import json
from pathlib import Path

import torch

from squid.checkpoints.store import load_checkpoint, load_model
from squid.device import get_device
from squid.eval.generate import generate_response_resumable
from squid.eval.judge import save_score_report, score_entries
from squid.tokenizer import TiktokenTokenizer
from squid.weight_edit import apply_scale_row_patch

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "weight_edit"
BASE_CHECKPOINT = ROOT / "data" / "model_pth" / "gpt2_355M_instruct.pth"
BASE_REPORT = ROOT / "data" / "instruct" / "m5-report-sft-20.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_patch_id(patch):
    content = json.dumps(patch, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def get_instruction_split(split):
    data = json.loads(
        (ROOT / "data" / "instruct" / "instruction-data.json").read_text(encoding="utf-8")
    )
    train_end = int(len(data) * 0.85)
    test_end = train_end + int(len(data) * 0.1)
    if split == "val":
        return data[test_end:]
    if split == "test":
        return data[train_end:test_end]
    raise ValueError(f"未知数据划分: {split}")


def generate_val_observations():
    tokenizer = TiktokenTokenizer("gpt2")
    model, _ = load_model(BASE_CHECKPOINT, get_device())
    val_data = get_instruction_split("val")[:20]
    responses_path = WORK / "validation-responses.json"
    responses = generate_response_resumable(
        model,
        tokenizer,
        val_data,
        responses_path,
    )
    return tokenizer, responses


def prepare():
    tokenizer, responses = generate_val_observations()
    checkpoint = load_checkpoint(BASE_CHECKPOINT)
    eos_row = checkpoint["model_state_dict"]["out_head.weight"][tokenizer.eos_id]
    stats = {
        "shape": list(eos_row.shape),
        "mean": eos_row.mean().item(),
        "std": eos_row.std().item(),
        "norm": eos_row.norm().item(),
        "min": eos_row.min().item(),
        "max": eos_row.max().item(),
    }
    prompt = f"""You are asked to propose exactly one bounded neural-network weight edit.

Goal: test whether an LLM can make a useful, auditable weight-space intervention on an SFT GPT-2 355M model.

Rules:
- Use only the validation observations below. Do not request or use held-out test answers.
- You may scale only the EOS row of out_head.weight.
- EOS token id: {tokenizer.eos_id}
- factor must be between 0.8 and 1.2.
- Propose exactly one patch. Do not generate executable code.
- If evidence is insufficient, factor 1.0 is an acceptable no-op control.
- Return only a JSON object with this schema:
{{
  "operation": "scale_row",
  "target": "out_head.weight",
  "row_index": {tokenizer.eos_id},
  "factor": 1.0,
  "rationale": "brief evidence-based explanation"
}}

EOS row statistics:
{json.dumps(stats, indent=2)}

Validation responses (these are validation, not test):
{json.dumps(responses, indent=2, ensure_ascii=False)}
"""
    prompt_path = WORK / "gpt-patch-prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"GPT 输入材料: {prompt_path}")
    print(f"验证集回答数: {len(responses)}")


def apply_patch(patch_path):
    patch = json.loads(Path(patch_path).read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(BASE_CHECKPOINT)
    edited_state = apply_scale_row_patch(
        checkpoint["model_state_dict"],
        patch,
        eos_id=50256,
    )
    patch_id = get_patch_id(patch)
    output_path = WORK / f"edited-{patch_id}.pth"
    edited_checkpoint = {
        **checkpoint,
        "model_state_dict": edited_state,
        "meta": {
            **checkpoint.get("meta", {}),
            "weight_edit": patch,
            "parent_checkpoint": str(BASE_CHECKPOINT),
            "parent_sha256": sha256_file(BASE_CHECKPOINT),
            "patch_id": patch_id,
        },
    }
    torch.save(edited_checkpoint, output_path)
    print(f"已保存权重副本: {output_path}")
    print(f"patch_id: {patch_id}")


def evaluate_patch(patch_path):
    patch = json.loads(Path(patch_path).read_text(encoding="utf-8"))
    patch_id = get_patch_id(patch)
    edited_checkpoint = WORK / f"edited-{patch_id}.pth"
    if not edited_checkpoint.exists():
        raise FileNotFoundError("先运行 apply 生成对应补丁权重")

    tokenizer = TiktokenTokenizer("gpt2")
    model, _ = load_model(edited_checkpoint, get_device())
    test_data = get_instruction_split("test")[:20]
    response_path = WORK / f"test-responses-{patch_id}.json"
    scored_path = WORK / f"test-scored-{patch_id}.json"
    report_path = WORK / f"test-report-{patch_id}.json"

    responses = generate_response_resumable(
        model,
        tokenizer,
        test_data,
        response_path,
    )
    scored = score_entries(
        responses,
        save_path=str(scored_path),
        flush_every=1,
    )
    edited_report = save_score_report(scored, report_path)
    baseline_report = json.loads(BASE_REPORT.read_text(encoding="utf-8"))

    comparison = {
        "patch": patch,
        "baseline_summary": baseline_report["summary"],
        "edited_summary": edited_report["summary"],
        "delta_mean": (
            edited_report["summary"]["mean"] - baseline_report["summary"]["mean"]
        ),
        "note": "LLM 评委重跑存在波动；小幅均分变化不足以证明有效，需检查逐题回答与分布。",
    }
    comparison_path = WORK / f"comparison-{patch_id}.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=4, ensure_ascii=False))
    print(f"对照报告: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="M5 受限权重编辑实验")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    for name in ["apply", "evaluate"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("patch", type=Path)
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare()
    elif args.command == "apply":
        apply_patch(args.patch)
    else:
        evaluate_patch(args.patch)


if __name__ == "__main__":
    main()
