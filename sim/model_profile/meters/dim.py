"""
weight & input/output dimension profiler
"""

import torch.nn as nn
from model_profile.meters.profiler import Profiler

def feature_hook(name, state_dict):
    def hook(model, input, output):
        state_dict[name] = [input[0].detach(), output.detach()]
    return hook

class DIM(Profiler):
    def __init__(self, name, model: nn.Module, device, input_size: int, precision=32) -> None:
        super().__init__(name, model, device, input_size, precision)

        # weight, input, output dimension
        self.w_dict = {}
        self.i_dict = {}
        self.o_dict = {}

    def hook(self):
        for n, m in self.model.named_modules():
            m.register_forward_hook(feature_hook(n, self.feature_dict))

    def conv_dim(self, layer:nn.Conv2d, name:str):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        
        # size
        bi, ci, hi, wi = i_feature.shape
        bo, co, ho, wo = o_feature.shape

        # kernel
        k = layer.kernel_size[0]
        cin = layer.in_channels // layer.groups

        self.i_dict[name] = [bi, wi, hi, ci]
        self.o_dict[name] = [bo, wo, ho, co]
        self.w_dict[name] = [k, k, cin, co]

    def linear_dim(self, layer:nn.Linear, name):
        self.i_dict[name] = [layer.in_features]
        self.o_dict[name] = [layer.out_features]
        self.w_dict[name] = [layer.in_features, layer.out_features]

    def fit(self):
        super().forward()

        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.conv_dim(m, n)
            elif isinstance(m, nn.Linear):
                self.linear_dim(m, n)
            else:
                print(f"{type(m)} will be ignored for dim calculation")


    