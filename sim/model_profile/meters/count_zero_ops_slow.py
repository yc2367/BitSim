"""
weight & input/output zero operations profiler
"""
import torch
import torch.nn as nn
from torchvision.models import resnet
from model_profile.meters.profiler import Profiler

def feature_hook(name, state_dict):
    def hook(model, input, output):
        state_dict[name] = [input[0].detach(), output.detach()]
    return hook

class CountZeroOps(Profiler):
    def __init__(self, name, model: nn.Module, device, input_size: int, precision=32) -> None:
        super().__init__(name, model, device, input_size, precision)  
        self.layer_name_list = []
        # number of zero operations for every layer
        self.num_zero_ops = {}

    def hook(self):
        for n, m in self.model.named_modules():
            m.register_forward_hook(feature_hook(n, self.feature_dict))
    
    def count_zero_ops_conv(self, layer: nn.Conv2d, name:str):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        # size
        bi, cin,  ih, iw = i_feature.shape
        bo, cout, oh, ow = o_feature.shape
        pad_h, pad_w = layer.padding
        assert bi == bo, 'ERROR! Batch size is not the same for input feature and output feature!'
        # kernel
        weight = layer.weight
        k   = layer.kernel_size[0]
        cin = layer.in_channels // layer.groups
        sh, sw = layer.stride

        # count number of zero operations for every output pixel
        i_feature = nn.functional.pad(i_feature, pad=[pad_w, pad_w, pad_h, pad_h])
        num_zero_ops = torch.zeros_like(o_feature)
        if cin == layer.in_channels: # conv2D
            for j_bi in range(bi): # batch dimension
                for j_cout in range(cout): # output channel dimension
                    kernel = weight[j_cout]
                    for j_oh in range(oh): # output height dimension
                        for j_ow in range(ow): # output width dimension
                            i_patch = i_feature[j_bi, :, j_oh*sh:j_oh*sh+k, j_ow*sw:j_ow*sw+k]
                            is_zero = ((i_patch * kernel) == 0)
                            num_zero_ops[j_bi, j_cout, j_oh, j_ow] = torch.sum(is_zero)
        else: # depthwise
            for j_bi in range(bi): # batch dimension
                for j_cout in range(cout): # output channel dimension
                    kernel = weight[j_cout, 0]
                    for j_oh in range(oh): # output height dimension
                        for j_ow in range(ow): # output width dimension
                            i_patch = i_feature[j_bi, j_cout, j_oh*sh:j_oh*sh+k, j_ow*sw:j_ow*sw+k]
                            is_zero = ((i_patch * kernel) == 0)
                            num_zero_ops[j_bi, j_cout, j_oh, j_ow] = torch.sum(is_zero)
        self.num_zero_ops[name] = torch.round(torch.mean(num_zero_ops, dim=0))
    
    def count_zero_ops_linear(self, layer: nn.Conv2d, name:str):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        # size
        bi, cin  = i_feature.shape
        bo, cout = o_feature.shape
        assert bi == bo, 'ERROR! Batch size is not the same for input feature and output feature!'
        # kernel
        weight = layer.weight

        # count number of zero operations for every output pixel
        num_zero_ops = torch.zeros_like(o_feature)
        for j_bi in range(bi): # batch dimension
            i_patch = i_feature[j_bi]
            for j_cout in range(cout): # output channel dimension
                kernel = weight[j_cout]
                is_zero = ((i_patch * kernel) == 0)
                num_zero_ops[j_bi, j_cout] = torch.sum(is_zero)
        self.num_zero_ops[name] = torch.round(torch.mean(num_zero_ops, dim=0))
        print(num_zero_ops)

    def fit(self):
        super().forward()

        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.count_zero_ops_conv(m ,n)
                self.layer_name_list.append(n)
            elif isinstance(m, nn.Linear):
                self.count_zero_ops_linear(m, n)
                self.layer_name_list.append(n)
