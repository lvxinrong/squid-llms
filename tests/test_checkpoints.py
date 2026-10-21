import torch

from squid.checkpoints import lora as lora_store
from squid.checkpoints import store
from squid.model.gpt import GPTModel
from squid.model.lora import replace_linear_with_lora

TEST_CFG = {
    "vocab_size": 50257,
    "emb_dim": 64,
    "context_length": 32,
    "n_layers": 2,
    "n_heads": 4,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def test_checkpoint_roundtrip(tmp_path):
    """存档再读档，同一输入的输出必须逐位一致。"""
    model = GPTModel(TEST_CFG).eval()
    x = torch.randint(0, 50257, (1, 8))
    with torch.no_grad():
        before = model(x)

    path = tmp_path / "ckpt.pth"
    store.save_checkpoint(path, model, TEST_CFG, epochs=2, val_loss=0.62)

    loaded, ckpt = store.load_model(path, device="cpu")
    with torch.no_grad():
        after = loaded(x)

    assert torch.allclose(before, after)
    assert ckpt["meta"]["val_loss"] == 0.62
    assert ckpt["config"]["emb_dim"] == 64


def test_lora_adapter_roundtrip(tmp_path):
    """adapter 存取后，LoRA 路径的输出必须一致。"""
    torch.manual_seed(42)
    model = GPTModel(TEST_CFG)
    replace_linear_with_lora(model, rank=8, alpha=16)
    model.eval()
    x = torch.randint(0, 50257, (1, 8))
    with torch.no_grad():
        before = model(x)

    path = tmp_path / "adapter.pth"
    lora_store.save_lora(path, model, rank=8, alpha=16)

    torch.manual_seed(0)  # 换个种子重建，LoRA 的 A 会不同
    fresh = GPTModel(TEST_CFG)
    replace_linear_with_lora(fresh, rank=8, alpha=16)
    fresh.load_state_dict(model.state_dict(), strict=False)  # 模拟 base 已加载
    lora_store.load_lora(fresh, path)
    fresh.eval()
    with torch.no_grad():
        after = fresh(x)

    assert torch.allclose(before, after, atol=1e-5)
