"""I3-2 INT4 Qwen 探针：将内部 Linear 逐行量化、两个 INT4 打包为一个 byte，并保留 out_head。

用途：测量真实模型的持久张量规模、内存和 logits 偏移，验证 packed INT4 能完成实际前向。
输入：真实 Qwen2.5-0.5B-Instruct；输出 data/reports/i3-int4-qwen-probe.json，含替换模块清单。
边界：共享权重通过 keep_vars=True 去重；当前为 W4A32 教学实现，不代表生产 INT4 kernel 的质量或性能。
"""

import json
from pathlib import Path

import torch

from squid.checkpoints.qwen import load_qwen2_5
from squid.device import get_device, memory_snapshot
from squid.generate.chat import build_qwen_chat_prompt
from squid.quantize.convert import quantize_model_int4
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = (
    ROOT
    / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
)
REPORT_PATH = ROOT / "data/reports/i3-int4-qwen-probe.json"


def state_dict_nbytes(model):
    total = 0
    seen = set()
    """
    model.parameters() 面向训练，遍历可训练参数；model.state_dict() 面向保存与恢复，获取模型的完整持久状态。
    model.parameters() 它返回一个参数迭代器，主要遍历模型中的 nn.Parameter
    
    model.state_dict() 它返回一个字典:
    {
        "模块路径.张量名称": tensor,
        ...
    }
    parameters() 回答“哪些张量需要训练”；state_dict() 回答“恢复这个模型需要保存哪些张量”。
    """
    for tensor in model.state_dict(keep_vars=True).values():
        tensor_id = id(tensor)
        if tensor_id in seen:
            continue

        seen.add(tensor_id)
        total += tensor.nelement() * tensor.element_size()

    return total


def main():
    device = get_device()
    model = load_qwen2_5(
        WEIGHTS_PATH,
        cfg={"context_length": 4096},
        device=device,
    )
    tokenizer = QwenTokenizerAdapter(
        "Qwen/Qwen2.5-0.5B-Instruct",
    )

    prompt = build_qwen_chat_prompt(
        tokenizer.tokenizer,
        [],
        "用一句话解释什么是权重量化。",
    )
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    bytes_before = state_dict_nbytes(model)
    memory_before = memory_snapshot(device)

    with torch.no_grad():
        reference_logits = model(idx)[0, -1, :].detach().cpu()

    replaced = quantize_model_int4(model)

    bytes_after = state_dict_nbytes(model)
    memory_after = memory_snapshot(device)

    with torch.no_grad():
        actual_logits = model(idx)[0, -1, :].detach().cpu()

    difference = actual_logits - reference_logits

    report = {
        "experiment": "i3-int4-qwen-probe",
        "model": "Qwen2.5-0.5B-Instruct",
        "device": str(device),
        "compute_dtype": str(next(model.parameters()).dtype),
        "weight_quantization": "per-row symmetric INT4, packed two values per byte",
        "activation_quantization": False,
        "out_head_quantized": False,
        "prompt_tokens": len(prompt_ids),
        "replaced_linear_count": len(replaced),
        "replaced_modules": replaced,
        "state_dict_bytes_before": bytes_before,
        "state_dict_bytes_after": bytes_after,
        "state_dict_reduction_percent": (
            (bytes_before - bytes_after) / bytes_before * 100
        ),
        "state_dict_compression_ratio": bytes_before / bytes_after,
        "max_abs_logit_difference": difference.abs().max().item(),
        "mean_abs_logit_difference": difference.abs().mean().item(),
        "reference_top1_token": reference_logits.argmax().item(),
        "actual_top1_token": actual_logits.argmax().item(),
        "top1_match": (
            reference_logits.argmax().item()
            == actual_logits.argmax().item()
        ),
        "memory_before_quantization": memory_before,
        "memory_after_quantization": memory_after,
        "out_head_remains_tied": (
            model.out_head.weight is model.tok_emb.weight
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    console_report = {
        key: value
        for key, value in report.items()
        if key != "replaced_modules"
    }
    print(json.dumps(console_report, ensure_ascii=False, indent=2))
    print(f"\nINT4 Qwen 探针报告已保存到：{REPORT_PATH}")

    assert len(replaced) == 168
    assert report["out_head_remains_tied"]
    assert torch.isfinite(actual_logits).all()


if __name__ == "__main__":
    main()
