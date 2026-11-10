import json
from pathlib import Path

from squid.checkpoints.store import load_checkpoint
from squid.model.v3_modern.gpt import GPTModelV3

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "regression/i1-memmap-smoke.pth"
TOKEN_METADATA_PATH = (
    ROOT / "data/token_store/blog_v1/metadata.json"
)
REPORT_PATH = ROOT / "data/reports/i1-scaling-budget.json"

def main():
    checkpoint = load_checkpoint(CHECKPOINT_PATH)

    model = GPTModelV3(checkpoint["config"])
    parameter_count = sum(
        parameter.numel() # number of elements  这个参数张量里一共有多少个元素
        for parameter in model.parameters()
    )

    for name, parameter in model.named_parameters():
        print(name, tuple(parameter.shape))

    token_metadata = json.loads(
        TOKEN_METADATA_PATH.read_text(encoding="utf-8")
    )
    corpus_tokens = token_metadata["splits"]["train"]["token_count"]

    smoke_reference_tokens = parameter_count * 20

    # 仅使用“0.5B 名义参数量”帮助理解数量级，
    # 不把它写成当前 Qwen 权重的精确参数统计。
    qwen_nominal_parameters = 500_000_000
    qwen_reference_tokens = qwen_nominal_parameters * 20

    report = {
        "rule": "约 20 tokens/参数，仅作从零预训练的近似参考",
        "corpus_train_tokens": corpus_tokens,
        "smoke_model": {
            "parameters": parameter_count,
            "reference_tokens": smoke_reference_tokens,
            "corpus_coverage": (
                corpus_tokens / smoke_reference_tokens
            ),
        },
        "qwen_0_5b_nominal_reference": {
            "parameters": qwen_nominal_parameters,
            "reference_tokens": qwen_reference_tokens,
            "corpus_coverage": (
                corpus_tokens / qwen_reference_tokens
            ),
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nScaling 报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()