import torch

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
    eager_model = GPTModelV3({
        **BASE_CONFIG,
        "attention_impl": "eager",
    })

    sdpa_model = GPTModelV3({
        **BASE_CONFIG,
        "attention_impl": "sdpa",
    })

    # 直接复制同一份权重，不依赖两次随机初始化恰好一致。
    sdpa_model.load_state_dict(eager_model.state_dict())

    eager_model.eval()
    sdpa_model.eval()
    return eager_model, sdpa_model


def assert_same_logits(eager_logits, sdpa_logits):
    torch.testing.assert_close(
        sdpa_logits,
        eager_logits,
        rtol=0.0,
        atol=1e-4,
    )


def test_sdpa_prefill_and_decode_match_eager():
    eager_model, sdpa_model = build_models()

    # B=1，T=8，使用同一段输入。
    idx = torch.randint(
        0,
        BASE_CONFIG["vocab_size"],
        (1, 8),
    )

    with torch.no_grad():
        # 场景一：整段输入，q_len>1，past_len=0，需要 causal mask。
        eager_full = eager_model(idx)
        sdpa_full = sdpa_model(idx)
        assert_same_logits(eager_full, sdpa_full)

        # 场景二：先对前 4 个 token 做 prefill。
        eager_logits, eager_past = eager_model(
            idx[:, :4],
            use_cache=True,
        )
        sdpa_logits, sdpa_past = sdpa_model(
            idx[:, :4],
            use_cache=True,
        )
        assert_same_logits(eager_logits, sdpa_logits)

        # 场景三：逐 token decode，q_len=1，past_len>0。
        for position in range(4, idx.shape[1]):
            next_token = idx[:, position:position + 1]

            eager_logits, eager_past = eager_model(
                next_token,
                past_kv=eager_past,
                use_cache=True,
            )
            sdpa_logits, sdpa_past = sdpa_model(
                next_token,
                past_kv=sdpa_past,
                use_cache=True,
            )

            assert_same_logits(eager_logits, sdpa_logits)
