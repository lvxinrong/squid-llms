"""I1 memmap 训练冒烟：验证流式 token 数据能完成训练、保存、重载和一致前向。

用途：用 5 个真实 optimizer step 检查数据管道与 checkpoint 闭环，而不是追求模型能力。
输入：data/token_store/blog_v1/train.uint32.bin；checkpoint 写入 regression/i1-memmap-smoke.pth。
边界：训练量极小；通过标准是 loss 有限、存档可重载且重载前后 logits 零误差，不代表模型已完成训练。
"""

from itertools import islice
from pathlib import Path

import torch

from squid.checkpoints.store import load_checkpoint, save_checkpoint
from squid.data.memmap_dataset import create_memmap_loader
from squid.device import get_device
from squid.model.v3_modern.gpt import GPTModelV3
from squid.train.common import calc_loss_batch


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = ROOT / "data/token_store/blog_v1/train.uint32.bin"
CHECKPOINT_PATH = ROOT / "regression/i1-memmap-smoke.pth"

def main():
    torch.manual_seed(123)
    device = get_device()

    cfg = {
        "vocab_size":151936,
        "emb_dim":64,
        "ffn_dim": 256,
        "n_heads": 4,
        "n_kv_heads":2,
        "n_layers":2,
        "rope_theta": 1000000.0,
        "eps": 1e-6,
        "qkv_bias": True,
        "ffn_bias": False,
        "tie_word_embeddings": False, #  Weight Tying（权重绑定/共享）self.out_head.weight = self.tok_emb.weight
        "drop_rate": 0.0,
        "context_length": 128,
    }

    train_loader = create_memmap_loader(
        TOKEN_PATH, max_length=128, batch_size=2, shuffle=True, drop_last=True
    )
    model = GPTModelV3(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 3e-4,
    )

    losses = []
    # enumerate 没看懂
    for step, (inputs, targets) in enumerate(
        islice(train_loader, 5),
        1
    ):
        optimizer.zero_grad()
        loss = calc_loss_batch(
            inputs,
            targets,
            model,
            device,
        )
        # 这句在做什么
        assert torch.isfinite(loss)

        loss.backward()
        optimizer.step()
        # loss.item的含义
        losses.append(loss.item())
        print(f"step {step}: loss={loss.item():.4f}")

    assert len(losses) == 5
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        CHECKPOINT_PATH,
        model,
        cfg,
        optimizer,
        steps=5,
        last_loss=losses[-1],
    )

    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    reloaded_model = GPTModelV3(checkpoint["config"])
    reloaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    reloaded_model.to(device)
    reloaded_model.eval()

    inputs, _ = next(iter(train_loader))
    model.eval()

    with torch.no_grad():
        expected_logits = model(inputs.to(device))
        actual_logits = reloaded_model(inputs.to(device))

    torch.testing.assert_close(
        actual_logits,
        expected_logits,
        rtol=0.0,
        atol=0.0,
    )

    print("\n5 步训练、保存、重载与前向一致性均通过")
    print(f"checkpoint：{CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()