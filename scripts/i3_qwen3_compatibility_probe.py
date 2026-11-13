"""I3 番外：读取 Qwen3-0.6B 权重元数据，验证 V4 所需结构假设。

用途：确认显式 head_dim、QK RMSNorm、GQA 投影形状和共享输出头的真实权重证据。
输入：本地 config.json 与 model.safetensors；输出 data/reports/i3-qwen3-compatibility.json。
边界：只检查 checkpoint 元数据，不证明模型前向、cache decode 或对话结果已经正确。
"""

import json
from pathlib import Path

from squid.checkpoints.inspect import inspect_safetensors


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"
WEIGHTS_PATH = MODEL_DIR / "model.safetensors"
REPORT_PATH = ROOT / "data/reports/i3-qwen3-compatibility.json"

def main():
    config = json.loads(
        (MODEL_DIR / "config.json").read_text(encoding="utf-8")
    )
    inspection = inspect_safetensors(WEIGHTS_PATH)
    by_name = {
        tensor["name"]: tensor
        for tensor in inspection["tensors"]
    }

    layer_zero = [
        tensor
        for tensor in inspection["tensors"]
        if tensor["name"].startswith("model.layers.0.")
    ]

    checks = {
        "model_type_is_qwen3": config["model_type"] == "qwen3",
        "explicit_head_dim_128": config["head_dim"] == 128,
        "q_norm_present": (
            "model.layers.0.self_attn.q_norm.weight" in by_name
        ),
        "k_norm_present": (
            "model.layers.0.self_attn.k_norm.weight" in by_name
        ),
        "q_norm_uses_head_dim": (
            by_name["model.layers.0.self_attn.q_norm.weight"]["shape"]
            == [128]
        ),
        "k_norm_uses_head_dim": (
            by_name["model.layers.0.self_attn.k_norm.weight"]["shape"]
            == [128]
        ),
        "q_proj_shape": (
            by_name["model.layers.0.self_attn.q_proj.weight"]["shape"]
            == [2048, 1024]
        ),
        "k_proj_shape": (
            by_name["model.layers.0.self_attn.k_proj.weight"]["shape"]
            == [1024, 1024]
        ),
        "v_proj_shape": (
            by_name["model.layers.0.self_attn.v_proj.weight"]["shape"]
            == [1024, 1024]
        ),
        "out_proj_shape": (
            by_name["model.layers.0.self_attn.o_proj.weight"]["shape"]
            == [1024, 2048]
        ),
        "no_independent_lm_head": "lm_head.weight" not in by_name,
        "embedding_shape": (
            by_name["model.embed_tokens.weight"]["shape"]
            == [config["vocab_size"], config["hidden_size"]]
        ),
    }

    report = {
        "model": "Qwen3-0.6B",
        "weight_file_bytes": inspection["file_bytes"],
        "weight_key_count": inspection["key_count"],
        "total_elements": inspection["total_elements"],
        "layer_zero_tensors": layer_zero,
        "checks": checks,
        "all_metadata_checks_passed": all(checks.values()),
        "boundary": "仅完成元数据兼容性审计，尚未执行模型前向或生成验收",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n兼容性报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()