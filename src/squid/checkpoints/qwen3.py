import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from squid.device import get_device
from squid.model.v4_qwen3.gpt import GPTModelV4


def map_qwen3_state_dict(state_dict, tie_word_embeddings):
    mapped = {}

    for key, value in state_dict.items():
        if key == "model.embed_tokens.weight":
            mapped["tok_emb.weight"] = value
            if tie_word_embeddings:
                mapped["out_head.weight"] = value
            continue

        if key == "lm_head.weight":
            if tie_word_embeddings:
                if not torch.equal(
                    value,
                    state_dict["model.embed_tokens.weight"],
                ):
                    raise ValueError(
                        "绑定模式下 lm_head 与 embedding 权重不一致"
                    )
            else:
                mapped["out_head.weight"] = value
            continue

        if key == "model.norm.weight":
            mapped["final_norm.scale"] = value
            continue

        if key == "model.rotary_emb.inv_freq":
            continue

        if not key.startswith("model.layers."):
            raise ValueError(f"未处理的 Qwen3 权重键: {key}")

        parts = key.split(".")
        layer = parts[2]
        suffix = ".".join(parts[3:])
        prefix = f"trf_block.{layer}"

        if suffix == "input_layernorm.weight":
            target = f"{prefix}.norm1.scale"
        elif suffix == "post_attention_layernorm.weight":
            target = f"{prefix}.norm2.scale"
        elif suffix == "self_attn.q_proj.weight":
            target = f"{prefix}.att.W_query.weight"
        elif suffix == "self_attn.q_proj.bias":
            target = f"{prefix}.att.W_query.bias"
        elif suffix == "self_attn.k_proj.weight":
            target = f"{prefix}.att.W_key.weight"
        elif suffix == "self_attn.k_proj.bias":
            target = f"{prefix}.att.W_key.bias"
        elif suffix == "self_attn.v_proj.weight":
            target = f"{prefix}.att.W_value.weight"
        elif suffix == "self_attn.v_proj.bias":
            target = f"{prefix}.att.W_value.bias"
        elif suffix == "self_attn.o_proj.weight":
            target = f"{prefix}.att.out_proj.weight"
        elif suffix == "self_attn.q_norm.weight":
            target = f"{prefix}.att.q_norm.scale"
        elif suffix == "self_attn.k_norm.weight":
            target = f"{prefix}.att.k_norm.scale"
        elif suffix == "mlp.gate_proj.weight":
            target = f"{prefix}.ffn.gate_proj.weight"
        elif suffix == "mlp.up_proj.weight":
            target = f"{prefix}.ffn.up_proj.weight"
        elif suffix == "mlp.down_proj.weight":
            target = f"{prefix}.ffn.down_proj.weight"
        else:
            raise ValueError(f"未处理的 Qwen3 权重键: {key}")

        mapped[target] = value

    return mapped


def load_qwen3(weights_path, cfg, device=None):
    state_dict = load_file(str(weights_path), device="cpu")
    mapped = map_qwen3_state_dict(
        state_dict,
        cfg["tie_word_embeddings"],
    )

    model = GPTModelV4(cfg)
    model.load_state_dict(mapped)
    model.to(device or get_device())
    model.eval()
    return model


def load_qwen3_dir(model_dir, context_size=4096, device=None):
    model_dir = Path(model_dir)
    config = json.loads(
        (model_dir / "config.json").read_text(encoding="utf-8")
    )

    if config.get("model_type") != "qwen3":
        raise ValueError("当前入口只支持 Qwen3 Dense")
    if config.get("hidden_act") != "silu":
        raise ValueError("当前实现要求 hidden_act=silu")
    if config.get("use_sliding_window", False):
        raise NotImplementedError("当前 V4 不支持 sliding attention")

    cfg = {
        "vocab_size": config["vocab_size"],
        "emb_dim": config["hidden_size"],
        "ffn_dim": config["intermediate_size"],
        "n_heads": config["num_attention_heads"],
        "n_kv_heads": config["num_key_value_heads"],
        "n_layers": config["num_hidden_layers"],
        "head_dim": config["head_dim"],
        "rope_theta": config.get("rope_theta", 10000.0),
        "eps": config.get("rms_norm_eps", 1e-6),
        "attention_bias": config.get("attention_bias", False),
        "attention_dropout": config.get("attention_dropout", 0.0),
        "drop_rate": 0.0,
        "context_length": context_size,
        "ffn_bias": False,
        "tie_word_embeddings": config.get(
            "tie_word_embeddings",
            False,
        ),
    }

    return load_qwen3(
        model_dir / "model.safetensors",
        cfg=cfg,
        device=device,
    )
