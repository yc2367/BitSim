"""
Vanilla to low precision modules
"""

import copy
import torch
import torch.nn as nn

from typing import Tuple
from src.module.base import _QBaseConv2d, _QBaseLinear
from src.module.attention import QAttention

from timm.models.vision_transformer import Attention

from src.quantization.adaround import AdaRound
from src.quantization.lsq import LSQ

weight_quantizer = {
    "adaround": AdaRound,
}

input_quantizer = {
    "lsq": LSQ
}

def get_parent_name(target:str) -> Tuple[str, str]:
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]

class Vanilla4Compress(object):
    def __init__(self, model:nn.Module, wbit:int=8, abit:int=8) -> None:
        self.model = model
        self.wbit = wbit
        self.abit = abit
        
        self.q_layers = ["blocks.10.attn", "blocks.11.attn"]

    def conv(self, layer:nn.Conv2d):
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
            wbit=self.wbit,
            abit=self.abit
        )
        
        # copy the weights and bias to the new layer
        new_layer.weight.data[:] = layer.weight
        
        if has_bias:
            new_layer.bias.data[:] = layer.bias

        return new_layer

    def linear(self, layer:nn.Linear):
        has_bias = layer.bias is not None
        
        new_layer = _QBaseLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=has_bias,
            wbit=self.wbit,
            abit=self.abit
        )

        new_layer.weight.data[:] = layer.weight
        
        if has_bias:
            new_layer.bias.data[:] = layer.bias
        return new_layer

    def attn(self, layer:Attention):
        qkv_bias = layer.qkv.bias is not None

        # initialize the attention block
        qattn = QAttention(
            dim=layer.qkv.in_features,
            num_heads=layer.num_heads,
            qkv_bias=qkv_bias,
            qk_norm=False,
            attn_drop=layer.attn_drop.p,
            proj_drop=layer.proj_drop.p
        )

        # conver the linear layer
        qqkv = self.linear(layer.qkv)
        qproj = self.linear(layer.proj)

        # assign the layer back
        setattr(qattn, "qkv", qqkv)
        setattr(qattn, "proj", qproj)
        return qattn
    
    def assign_quantizer(self, model, wqtype, xqtype):
        model = copy.deepcopy(model)
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, (_QBaseConv2d, _QBaseLinear)):
                parent_name, name = get_parent_name(n)

                m.wq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=True, weights=m.weight)
                m.aq = input_quantizer[xqtype](nbit=self.abit, train_flag=True)

                setattr(modules[parent_name], name, m)

        return model
    
    def convert(self):
        model = copy.deepcopy(self.model)
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                parent_name, name = get_parent_name(n)

                if isinstance(m, nn.Conv2d):
                    new_layer = self.conv(m)
                    setattr(modules[parent_name], name, new_layer)
                elif isinstance(m, nn.Linear):
                    new_layer = self.linear(m)
                    setattr(modules[parent_name], name, new_layer)

        return model
    
    def reload(self, wqtype, xqtype):
        qmodel = self.convert()
        qmodel = self.assign_quantizer(qmodel, wqtype=wqtype, xqtype=xqtype)
        return qmodel
