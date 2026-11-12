from torch import nn

from squid.quantize.linear import QuantizedLinearInt8
from squid.quantize.linear_int4 import QuantizedLinearInt4


def quantize_model(model, linear_class, skip_names=None):
    if skip_names is None:
        skip_names = {"out_head"}
    else:
        skip_names = set(skip_names)

    replaced = []

    def replace_children(parent, prefix=""):
        for name, child in parent.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if full_name in skip_names:
                continue

            if isinstance(child, nn.Linear):
                quantized = linear_class.from_float(child)
                setattr(parent, name, quantized)
                replaced.append(full_name)
            else:
                replace_children(child, full_name)

    replace_children(model)
    return replaced


def quantize_model_int8(model, skip_names=None):
    return quantize_model(
        model,
        QuantizedLinearInt8,
        skip_names,
    )


def quantize_model_int4(model, skip_names=None):
    return quantize_model(
        model,
        QuantizedLinearInt4,
        skip_names,
    )
