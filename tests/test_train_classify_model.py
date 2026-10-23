import pandas as pd
import torch

from squid.data.classify import ClassifyDataset, classify_collate
from squid.device import get_device
from squid.model.gpt import GPTModel
from squid.tokenizer import TiktokenTokenizer
from squid.train.classify import prepare_classify_model, train_classifier_model

TEST_CFG = {
    "vocab_size": 50257,
    "context_length": 32,
    "emb_dim": 32,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def _fake_df():
    # 带明确信号词的二分类玩具数据：含 good→1，含 bad→0
    rows = []
    for i in range(20):
        rows.append(("ham", f"this is good product number {i}"))
        rows.append(("spam", f"this is bad offer number {i}"))
    return pd.DataFrame(rows, columns=["Label", "Text"])


def test_train_classify_model():
    device = get_device()
    tokenizer = TiktokenTokenizer("gpt2")
    df = _fake_df()

    ds = ClassifyDataset(df, tokenizer, max_length=16)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=True,
        collate_fn=lambda b: classify_collate(b, pad_id=tokenizer.eos_id),
    )

    model = GPTModel(TEST_CFG)
    prepare_classify_model(model, num_classes=2, emb_dim=TEST_CFG["emb_dim"])
    model.to(device)

    # 验收第一关：冻结生效，可训练参数远小于总量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\ntrainable {trainable:,} / total {total:,}")
    assert trainable < total * 0.5
    # 新头必须可训练且形状正确
    assert model.out_head.out_features == 2
    assert model.out_head.weight.requires_grad

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=5e-4
    )
    train_losses, val_losses, train_accs, val_accs, _ = train_classifier_model(
        model, loader, loader, optimizer, device,
        num_epochs=2, eval_freq=5, eval_iter=2,
    )

    # 验收第二关：loss 下降
    assert len(train_losses) >= 2
    assert train_losses[-1] < train_losses[0]
    # 验收第三关：accuracy 被计算且在合法范围
    assert 0.0 <= train_accs[-1] <= 1.0
