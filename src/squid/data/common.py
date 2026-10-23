import torch
import torch.nn.functional as F


def pad_to_length(seq: torch.Tensor, max_len: int, pad_id: int) -> torch.Tensor:
    # 右侧补 pad_id 到 max_len；够长了就原样返回
    pad_len = max_len - len(seq)
    if pad_len <= 0:
        return seq
    return F.pad(seq, (0, pad_len), value=pad_id)