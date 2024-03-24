"""
FLOPs profiler
"""

import torch
import torch.nn as nn
from model_profile.meters.profiler import Profiler
from model_profile.meters.utils import count_adap_avgpool

def feature_hook(name, state_dict):
    def hook(model, input, output):
        state_dict[name] = [input[0].detach(), output.detach()]
    return hook

class OPS(Profiler):
    def __init__(self, name, model: nn.Module, device, input_size: int, precision=32) -> None:
        super().__init__(name, model, device, input_size, precision)

        # total operations
        self.ops_dict = {}

    def hook(self):
        for n, m in self.model.named_modules():
            m.register_forward_hook(feature_hook(n, self.feature_dict))

    def conv_ops(self, layer:nn.Conv2d, name:str):
        feature = self.feature_dict[name][1]
        
        # size
        b, co, ho, wo = feature.shape

        # kernel
        k = layer.kernel_size[0]
        cin = layer.in_channels // layer.groups

        ops = (k**2) * co * ho * wo * cin
        self.ops_dict[name] = ops

    def bn_ops(self, layer:nn.BatchNorm2d, name):
        feature = self.feature_dict[name][1]

        ops = feature[0].numel() * 2
        if layer.affine:
            ops *= 2
        
        self.ops_dict[name] = ops

    def linear_ops(self, layer:nn.Linear, name):
        feature = self.feature_dict[name][1]

        weight_ops = layer.in_features * feature.numel()
        bias_ops = layer.bias.numel()

        ops = weight_ops + bias_ops
        self.ops_dict[name] = ops

    def adp_avg_pool_ops(self, layer:nn.AdaptiveAvgPool2d, name):
        x, y = self.feature_dict[name]

        ops = count_adap_avgpool(x, y)
        self.ops_dict[name] = ops.item()

    def avg_pool_ops(self, layer:nn.AvgPool2d, name):
        x, y = self.feature_dict[name]

        ops = y.numel()
        self.ops_dict[name] = ops

    def fit(self):
        super().forward()

        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.conv_ops(m, n)
            elif isinstance(m, nn.Linear):
                self.linear_ops(m, n)
            elif isinstance(m, nn.BatchNorm2d):
                self.bn_ops(m, n)
            elif isinstance(m, nn.AvgPool2d):
                self.avg_pool_ops(m, n)
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                self.adp_avg_pool_ops(m, n)
            else:
                print(f"{type(m)} will be ignored for OPS")

    