"""
Low precision MLP
"""

import torch.nn as nn
from timm.layers.mlp import Mlp

class QMlp(Mlp):
    def __init__(
            self, 
            in_features, 
            hidden_features=None, 
            out_features=None, 
            act_layer=nn.GELU, 
            norm_layer=None, 
            bias=True, 
            drop=0, 
            use_conv=False
        ):
        super().__init__(in_features, hidden_features, out_features, act_layer, norm_layer, bias, drop, use_conv)

    def forward(self, x):
        return super().forward(x)
    
