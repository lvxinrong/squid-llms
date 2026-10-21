"""LoRA adapter 存档：只存 A、B 和配置，几 MB，不存冻结的主体。

组合顺序（先结构后权重）：
    model = GPTModel(config)
    replace_linear_with_lora(model, rank, alpha)   # 先包结构
    load base 权重                                  # 再走 store.load
    load_lora(model, adapter_path)                 # 最后贴 adapter
"""

import torch

from squid.model.lora import LinearWithLoRA


def save_lora(path, model, rank, alpha, **meta):
    adapter = {}
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            adapter[f"{name}.A"] = module.lora.A.data
            adapter[f"{name}.B"] = module.lora.B.data
    torch.save({"rank": rank, "alpha": alpha, "adapter": adapter, "meta": meta}, path)
    return path


def load_lora(model, path, strict=True):
    """把 adapter 贴回已经包好 LoRA 结构的模型。strict=True 时键数必须全对上。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    adapter = payload["adapter"]

    found = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LinearWithLoRA):
                a_key, b_key = f"{name}.A", f"{name}.B"
                if a_key in adapter:
                    module.lora.A.data.copy_(adapter[a_key])
                    module.lora.B.data.copy_(adapter[b_key])
                    found += 1
                elif strict:
                    raise KeyError(f"adapter 里缺少 {a_key}/{b_key}")
    if strict and found * 2 != len(adapter):
        raise KeyError(f"模型只消化了 {found * 2} 个张量，adapter 里有 {len(adapter)} 个")
    return payload
