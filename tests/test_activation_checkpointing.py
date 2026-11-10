import torch
import torch.nn.functional as F

from squid.model.v3_modern.gpt import GPTModelV3


BASE_CONFIG = {
    "vocab_size": 128,
    "emb_dim": 16,
    "ffn_dim": 32,
    "n_heads": 4,
    "n_kv_heads": 2,
    "n_layers": 2,
    "rope_theta": 1000000.0,
    "eps": 1e-6,
    "qkv_bias": True,
    "ffn_bias": False,
    "tie_word_embeddings": False,
    "drop_rate": 0.0,
    "context_length": 16,
}

def build_models():
    standard_model = GPTModelV3({
        **BASE_CONFIG,
        "activation_checkpointing": False,
    })

    checkpoint_model = GPTModelV3({
        **BASE_CONFIG,
        "activation_checkpointing": True,
    })

    # 直接复制同一份权重，排除初始化差异。
    checkpoint_model.load_state_dict(
        standard_model.state_dict()
    )

    # checkpoint 只在训练状态生效。
    standard_model.train()
    checkpoint_model.train()

    return standard_model, checkpoint_model

def test_activation_checkpointing_logits_and_gradients():
    standard_model, checkpoint_model = build_models()
    idx = torch.randint(
        0,
        BASE_CONFIG["vocab_size"],
        (1, 8),
    )
    targets = torch.randint(
        0,
        BASE_CONFIG["vocab_size"],
        (1, 8),
    )

    standard_logits = standard_model(idx)
    checkpoint_logits = checkpoint_model(idx)

    # 前向输出应保持一致。
    torch.testing.assert_close(
        checkpoint_logits,
        standard_logits,
        rtol=0.0,
        atol=1e-4,
    )

    standard_loss = F.cross_entropy(
        standard_logits.flatten(0, 1),
        targets.flatten(),
    )

    checkpoint_loss = F.cross_entropy(
        checkpoint_logits.flatten(0, 1),
        targets.flatten(),
    )

    torch.testing.assert_close(
        checkpoint_loss,
        standard_loss,
        rtol=0.0,
        atol=1e-4,
    )

    standard_loss.backward()
    checkpoint_loss.backward()

    standard_params = dict(standard_model.named_parameters())
    checkpoint_params = dict(checkpoint_model.named_parameters())

    assert standard_params.keys() == checkpoint_params.keys()

    for name in standard_params:
        standard_grad = standard_params[name].grad
        checkpoint_grad = checkpoint_params[name].grad

        assert standard_grad is not None
        assert checkpoint_grad is not None

        torch.testing.assert_close(
            checkpoint_grad,
            standard_grad,
            rtol=1e-4,
            atol=1e-4,
            msg=f"{name} 的梯度不一致",
        )

