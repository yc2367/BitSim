"""
Decimal to binary
"""
import sys
sys.path.append("../../")
import torch.nn as nn
from src.d2c.util import *
from src.module.fuse import QConvReLU, QConvBNReLU
from src.module.base import _QBaseLinear

from software.util.bitflip_layer import bitflip_zeroPoint_conv, bitflip_zeroPoint_fc

class D2C(object):
    def __init__(self, model:nn.Module, wbit:int, args):
        self.model = model
        self.wbit = wbit
        self.grp_size = args.grp_size

        self.pruned_column_num = args.N
        self.func = args.flag   # 0 for signed magnitude, 1 for 2s complement

        self.hamming_distance = args.hamming_distance

        # fetch weights
        self.target_layers = ["layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"]
        

    def fetch_weights(self):
        self.weight_dict = {}

        for n, m in self.model.named_modules():
            if isinstance(m, (QConvBNReLU, QConvReLU)):
                if n in self.target_layers:
                    self.weight_dict[n] = m.conv.weight.data.detach()
            elif isinstance(m, _QBaseLinear):
                print(n)
                if n in self.target_layers:
                    self.weight_dict[n] = m.weight.data.detach()

    
    def convert(self):
        
        self.fetch_weights()

        self.new_weights = {}
        for k, weight in self.weight_dict.items():
            print(f"Layer [{k}] Start Conversion!")
            if self.func == 0:
                if len(weight.shape) == 4:
                    weight_new = process_signMagnitude_conv(weight, w_bitwidth=self.wbit, group_size=self.grp_size, 
                                                                    pruned_column_num=self.pruned_column_num, device=weight.device)
                    
                elif len(weight.shape) == 2:
                    weight_new = process_signMagnitude_fc(weight, w_bitwidth=self.wbit, group_size=self.grp_size, 
                                                                    pruned_column_num=self.pruned_column_num, device=weight.device)

            else:
                if len(weight.shape) == 4:
                    weight_new = bitflip_zeroPoint_conv(weight, w_bitwidth=self.wbit, group_size=self.grp_size, num_pruned_column=self.pruned_column_num, device="cpu")

                if len(weight.shape) == 2:
                    weight_new = bitflip_zeroPoint_fc(weight, w_bitwidth=self.wbit, group_size=self.grp_size, num_pruned_column=self.pruned_column_num, device="cpu")

            # update the dictionary
            mse_error = torch.nn.functional.mse_loss(weight_new, weight)
            print(f"MSE Error = {mse_error}")
            self.new_weights[k] = weight_new

        print("Conversion completed!")
    
    def reload(self):
        for n, m in self.model.named_modules():
            if isinstance(m, (QConvBNReLU, QConvReLU)):
                if n in self.target_layers:
                    m.conv.weight.data.copy_(self.new_weights[n])
            
            elif isinstance(m, _QBaseLinear):
                if n in self.target_layers:
                    m.weight.data.copy_(self.new_weights[n])
        
        print("Reload completed!")
        return self.model

    def fit(self):
        self.convert()
        model = self.reload()
        return model