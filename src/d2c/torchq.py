"""
Bit conversion for quantized pytorch model
"""

import torch.nn as nn
from src.d2c.util import *
from src.d2c.base import D2C

class QD2C(D2C):
    def __init__(self, model: nn.Module, wbit: int, args):
        super().__init__(model, wbit, args)

    def fetch_weights(self):
        self.weight_dict = {}
        self.bias_dict = {}
        
        self.scale_dict = {}
        self.zero_point_dict = {}

        for n, m in self.model.named_modules():
            if hasattr(m, "weight"):
                if n in self.target_layers:
                    w = m.weight()
                    
                    wint = torch.int_repr(w)
                    scale = w.q_per_channel_scales()
                    zp = w.q_per_channel_zero_points() 
                    
                    self.weight_dict[n] = wint.float()
                    self.bias_dict[n] = m.bias()
                    
                    self.scale_dict[n] = scale
                    self.zero_point_dict[n] = zp


    def reload(self):
        for n, m in self.model.named_modules():
            if hasattr(m, "weight"):
                if n in self.target_layers:
                    # wint = self.weight_dict[n]
                    wint = self.new_weights[n]
                    scale = self.scale_dict[n]
                    zp = self.zero_point_dict[n]

                    if "fc" in n:
                        wdq = wint.sub(zp[:, None]).mul(scale[:, None])
                        wint = torch.quantize_per_channel(wdq.float(), scale, zp, axis=0, dtype=torch.qint8)

                        m._packed_params.set_weight_bias(wint, self.bias_dict[n])
                    else: 
                        wdq = wint.sub(zp[:, None, None, None]).mul(scale[:, None, None, None])
                        wint = torch.quantize_per_channel(wdq.float(), scale, zp, axis=0, dtype=torch.qint8)

                        m.set_weight_bias(wint, self.bias_dict[n])
                    

        print("Reload completed!")
        return self.model
        
    def fit(self):
        self.convert()
        model = self.reload()
        return model