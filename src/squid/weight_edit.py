import math


def validate_scale_row_patch(patch, eos_id):
    required = {"operation", "target", "row_index", "factor"}
    missing = required - patch.keys()
    if missing:
        raise ValueError(f"权重补丁缺少字段: {sorted(missing)}")
    if patch["operation"] != "scale_row":
        raise ValueError("仅支持 scale_row")
    if patch["target"] != "out_head.weight":
        raise ValueError("本实验只允许修改 out_head.weight")

    row_index = patch["row_index"]
    factor = patch["factor"]
    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise ValueError("row_index 必须是整数")
    if row_index != eos_id:
        raise ValueError(f"本实验只允许修改 EOS 行: {eos_id}")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ValueError("factor 必须是数值")
    if not math.isfinite(factor) or not 0.8 <= factor <= 1.2:
        raise ValueError("factor 必须位于 [0.8, 1.2]")
    return patch


def apply_scale_row_patch(state_dict, patch, eos_id):
    validate_scale_row_patch(patch, eos_id)
    target = patch["target"]
    if target not in state_dict:
        raise KeyError(f"模型中不存在 {target}")

    source = state_dict[target]
    row_index = patch["row_index"]
    if row_index >= source.shape[0]:
        raise IndexError("row_index 超出目标张量范围")

    edited = source.clone()
    edited[row_index].mul_(patch["factor"])

    new_state = state_dict.copy()
    new_state[target] = edited
    return new_state
