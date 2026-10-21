import torch
from squid.model.gpt import GPTModel


def load_pretrained_gpt(config, weights_path="gpt2_335M.pth", device="cpu"):
    model = GPTModel(config)
    # 先加载到 CPU（稳定、兼容 weights_only），再整体搬到目标设备
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = {
        k.replace("trf_blocks.", "trf_block.").replace(".ff.", ".ffn."): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_instruct_checkpoint(path, config, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GPTModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint