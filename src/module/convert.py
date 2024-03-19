"""
Convert the vanilla module into the compression-compatible modules
"""

import copy
import torch
import torch.nn as nn
from typing import Tuple, Union

from src.module.base import _QBaseConv2d, _QBaseLinear

def get_parent_name(target:str) -> Tuple[str, str]:
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]

def fp2q(layer:Union[nn.Linear, nn.Conv2d], wbit:int, abit:int):
    r"""
    Convert the vanilla module (Convolution, Linear, Attention) into compression-ready modules.
    """
    with torch.no_grad():
        layer = copy.deepcopy(layer)

        if isinstance(layer, nn.Conv2d):
            has_bias = layer.bias is not None

            new_layer = _QBaseConv2d(
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                bias = has_bias,
                wbit=wbit,
                abit=abit
            )
            
            # copy the weights and bias to the new layer
            new_layer.weight.data[:] = layer.weight
            
            if has_bias:
                new_layer.bias.data[:] = layer.bias

        if isinstance(layer, nn.Linear):
            new_layer = _QBaseLinear(
                layer.in_features,
                layer.out_features,
                bias=True,
                wbit=wbit,
                abit=abit
            )

            # copy the weights and bias to the new layer
            new_layer.weight.data[:] = layer.weight

            if layer.bias is not None:
                new_layer.bias.data[:] = layer.bias
            
    
    return new_layer

def convert_model(model:nn.Module, wbit:int, abit:int):

    model = copy.deepcopy(model)
    modules = dict(model.named_modules(remove_duplicate=True))
    
    for n, m in modules.items():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            new_layer = fp2q(m, wbit=wbit, abit=abit)
            parent_name, name = get_parent_name(n)
            setattr(modules[parent_name], name, new_layer)

    return model

