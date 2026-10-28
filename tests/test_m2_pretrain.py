"""M2 收官：真实语料 + 自训 BPE 的完整预训练。

链路：pack_tokens 打包 → create_stream_loader → 小 GPT 预训练
      → bpb 曲线 → generate_text 咿呀学语 → save_checkpoint 带 config。
"""
import math
from pathlib import Path

import torch

from squid.data.token_stream import pack_tokens, create_stream_loader
from squid.device import get_device
from squid.generate.text import generate_text
from squid.model.gpt import GPTModel
from squid.train.common import calc_loss_loader
from squid.train.pretrain import pretrain_model
from squid.tokenizer.my_bpe_tokenizer import MyBPETokenizer
from squid.checkpoints.store import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "blog"


def test_m2_pretrain():
    device = get_device()
    tokenizer = MyBPETokenizer.from_directory(DATA / "m2000")

    # 1. token 流打包：训练语料与 held-out 分别打包
    train_text = (DATA / "train.txt").read_text(encoding="utf-8")
    val_text = "\n".join(p.read_text(encoding="utf-8") for p in DATA.glob("heldout-*.txt"))

    train_ids = pack_tokens([train_text], tokenizer, tokenizer.eos_id)
    val_ids = pack_tokens([val_text], tokenizer, tokenizer.eos_id)

    context = 128
    train_loader = create_stream_loader(train_ids, context, batch_size=8, shuffle=True)
    val_loader = create_stream_loader(val_ids, context, batch_size=8,
                                      shuffle=False, drop_last=False)

    # 2. 小模型：emb_dim 64、2 层、context 128
    cfg = {
        "vocab_size": len(tokenizer.vocab),
        "context_length": context,
        "emb_dim": 256,
        "n_heads": 2,
        "n_layers": 4,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 3. 预训练：accum_step=4，bpb 用 held-out 的 bytes/token 折算
    density = len(val_text.encode("utf-8")) / len(tokenizer.encode(val_text))
    train_losses, val_losses, tokens_seen = pretrain_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=20, eval_freq=50, eval_iter=5,
        accum_step=4, bpb_density=density,
    )
    assert val_losses[-1] < val_losses[0], "val loss 应该下降"

    # 4. 咿呀学语：给模型一个开头，看它续什么
    sample = generate_text(model, tokenizer, "大模型",
                           max_new_tokens=40, context_size=context,
                           temperature=0.8, top_k=20, eos_id=tokenizer.eos_id)
    print("\n=== 生成样例 ===\n", sample)

    # 5. 存档：权重 + config + 元信息，完整还原训练现场
    final_val = calc_loss_loader(val_loader, model, device)
    save_checkpoint(ROOT / "regression" / "m2-pretrain.pth",
                    model, cfg, val_loss=final_val, merges=2000, tokens_seen=tokens_seen[-1])
    print(f"\n最终 val_loss={final_val:.3f}，checkpoint 已保存")
