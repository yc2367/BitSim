"""
weight & input/output dimension profiler
"""

import torch.nn as nn
from torchvision.models import resnet
from model_profile.meters.profiler import Profiler

def feature_hook(name, state_dict):
    def hook(model, input, output):
        state_dict[name] = [input[0].detach(), output.detach()]
    return hook

class DIM(Profiler):
    def __init__(self, name, model: nn.Module, device, input_size: int, precision=32) -> None:
        super().__init__(name, model, device, input_size, precision)  
        self.layer_name_list = []

        # weight, input, output dimension
        self.weight_dim = {}
        self.input_dim = {}
        self.output_dim = {}

    def hook(self):
        for n, m in self.model.named_modules():
            m.register_forward_hook(feature_hook(n, self.feature_dict))

    def conv_dim(self, layer: nn.Conv2d, name:str):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        
        # size
        bi, ci, hi, wi = i_feature.shape
        bo, co, ho, wo = o_feature.shape

        # kernel
        k = layer.kernel_size[0]
        cin = layer.in_channels // layer.groups

        self.input_dim[name] = [bi, wi, hi, ci]
        self.output_dim[name] = [bo, ho, wo, co]
        self.weight_dim[name] = [k, k, cin, co]

    def linear_dim(self, layer: nn.Linear, name):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        if len(i_feature.shape) == 2: # CNN
            bi, ci = i_feature.shape
            bo, co = o_feature.shape
            self.input_dim[name] = [bi, 1, ci]
            self.output_dim[name] = [bo, 1, co]
            self.weight_dim[name] = [ci, co]
        elif len(i_feature.shape) == 3: # transformer
            bi, si, ci = i_feature.shape
            bo, so, co = o_feature.shape
            self.input_dim[name] = [bi, si, ci]
            self.output_dim[name] = [bo, so, co]
            self.weight_dim[name] = [ci, co]
        else:
            raise Exception('ERROR! More than 3 dimensions is provided for linear layer!')

    def bottleneck_dim(self, layer: resnet.Bottleneck, name):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        # size
        bi, ci, hi, wi = i_feature.shape
        bo, co, ho, wo = o_feature.shape

        self.weight_dim[name] = None
        self.input_dim[name] = [bi, wi, hi, ci]
        self.output_dim[name] = [bo, ho, wo, co]

    def fit(self):
        super().forward()

        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.conv_dim(m, n)
                self.layer_name_list.append(n)
            elif isinstance(m, nn.Linear):
                self.linear_dim(m, n)
                self.layer_name_list.append(n)
            '''
            elif isinstance(m, resnet.Bottleneck):
                self.bottleneck_dim(m, n)
                self.layer_name_list.append(n)
            else:
                print(f"{type(m)} will be ignored for dim calculation")
            ''' 