import torch
from setuptools.command.bdist_egg import walk_egg
from torch import nn
from torch.nn import functional as F

from squid.quantize.weight import (
    dequantize_weight_int8,
    quantize_weight_int8,
)

"""
    训练阶段：
        FP32 权重是需要学习的参数
        → 使用 nn.Parameter
        → 参与梯度计算
        → 由 optimizer 更新
    训练完成后做 PTQ：
        INT8 权重和 scale 已经确定
        → 不再学习
        → 使用 register_buffer 管理
        → 只参与推理、设备迁移和状态保存
    
    这些 INT8 权重和 scale 已经成为固定状态，不再需要通过梯度学习。buffer 同时保证了设备迁移和 state_dict 保存。
"""
class QuantizedLinearInt8(nn.Module):
    def __init__(self, quantized_weight, weight_scale, bias = None):
        super(QuantizedLinearInt8, self).__init__()
        # in_features  = 每个输入位置有多少特征，也是权重每行有多少列
        # out_features = 每个输出位置有多少特征，也是权重矩阵有多少行
        self.in_features = quantized_weight.shape[1]
        self.out_features = quantized_weight.shape[0]

        self.register_buffer("quantized_weight", quantized_weight)

        self.register_buffer("weight_scale", weight_scale)

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)


    @classmethod
    def from_float(cls, linear):
        quantized_weight, quantized_scale = quantize_weight_int8(
            # detach() 断开自动求导关系 复制转换时，不再追踪它原本参与的自动求导关系。
            linear.weight.detach()
        )

        bias = (
            # detach() 管计算图，clone() 管数据存储；前者不参与梯度，后者不与原张量共用数据。
            linear.bias.detach().clone()
            if linear.bias is not None
            else None
        )

        return cls(quantized_weight, quantized_scale, bias = bias)

    def forward(self, x):
        weight = dequantize_weight_int8(self.quantized_weight, self.weight_scale,)
        return F.linear(x, weight, self.bias)