"""I1 BPB 评估：用 memmap held-out 数据衡量 smoke 模型的文本预测 / 压缩能力。

用途：将交叉熵从 nats/token 换算为 bits/byte，使损失能与原始文本字节数关联。
输入：regression/i1-memmap-smoke.pth 和 heldout.uint32.bin；输出 data/reports/i1-bpb-smoke.json。
边界：累计所有 target 的 nats 后再平均，避免小批次等权误差；EOS 不计入自然文本字节，结果仅对应当前 checkpoint 与语料。
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from squid.checkpoints.store import load_checkpoint
from squid.data.memmap_dataset import create_memmap_loader
from squid.device import get_device
from squid.model.v3_modern.gpt import GPTModelV3
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "regression/i1-memmap-smoke.pth"
TOKEN_PATH = ROOT / "data/token_store/blog_v1/heldout.uint32.bin"
REPORT_PATH = ROOT / "data/reports/i1-bpb-smoke.json"


def main():
    device = get_device()
    checkpoint = load_checkpoint(CHECKPOINT_PATH)

    model = GPTModelV3(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = QwenTokenizerAdapter()

    loader = create_memmap_loader(
        TOKEN_PATH,
        max_length=128,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    total_nats = 0.0
    total_tokens = 0
    total_bytes = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            # 先累计整批 nats，最后再除以总 token 数，
            # 避免最后一个较小 batch 被平均成相同权重。
            batch_nats = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            )

            total_nats += batch_nats.item()
            total_tokens += targets.numel()

            # 只统计实际参与评估的 target 文本字节；
            # EOS 不折算成自然文本字节。
            target_text = tokenizer.decode(
                targets.flatten().tolist(),
                skip_special_tokens=True,
            )
            total_bytes += len(target_text.encode("utf-8"))

    mean_loss = total_nats / total_tokens

    # nats/token → bits/token → bits/byte
    # 模型对文本的预测/压缩能力。预测越准确，理论上编码这段文本所需的 bits 就越少，所以 BPB 越低越好。
    # loss 是模型预测代价，BPB 是把这个代价换算到每个原始字节上的结果。
    bpb = total_nats / math.log(2) / total_bytes

    report = {
        "checkpoint": str(CHECKPOINT_PATH),
        "evaluated_tokens": total_tokens,
        "evaluated_utf8_bytes": total_bytes,
        "mean_loss": mean_loss,
        "bpb": bpb,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nBPB 报告已保存到：{REPORT_PATH}")


if __name__ == "__main__":
    main()
