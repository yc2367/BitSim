"""
BatchNorm fusion with full observability
"""

import torch
import torch.nn as nn
from src.module.fuse import QConvReLU, QConvBNReLU, _QBaseConv2d, _QBaseLinear, FusedLinear, MulQuant
from typing import List

class LayerFuser(object):
    def __init__(self, model:nn.Module):
        self.model = model
        # flag
        self.flag = False
        
        # layers
        self.groups = []
        
        # parameters
        self.xscales = []
        self.xzps = []

        # full precision conv layer
        self.fpl = 1

        # full precision classifier
        self.fpc = False
    
    def inference(self):
        """
        Switch to inference mode
        """
        for n, m in self.model.named_modules():
            if hasattr(m, "inference"):
                m.inference()

    def layers(self):
        """
        Fetch layer information from pretrained model
        """
        conv_bn_relu = []
        l = 0
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, "wbit"):
                self.fpl += 1
            
            elif isinstance(m, _QBaseConv2d):
                self.flag = True
                conv_bn_relu.append(m)

                # scales and boundaries
                self.xscales.append(m.aq.scale.data)
                self.xzps.append(m.aq.zero_point.data)
                l += 1
            
            elif isinstance(m, nn.BatchNorm2d) and self.flag:
                conv_bn_relu.append(m)
            
            elif isinstance(m, nn.ReLU) and self.flag:
                conv_bn_relu.append(m)
                self.groups.append(conv_bn_relu)
                
                # reset
                self.flag = False
                conv_bn_relu = []
            
            elif isinstance(m, _QBaseLinear):
                self.fpc = False
                
                if not isinstance(m.aq, nn.Identity):
                    # scales and boundaries
                    self.xscales.append(m.aq.scale.data)
                    self.xzps.append(m.aq.zero_point.data)
                    l += 1

    def fuse_linear(self, layer:_QBaseLinear):
        tmp = FusedLinear(layer.in_features, layer.out_features, True, wbit=layer.wbit, abit=layer.abit, train_flag=False)
        
        # fetch the post-computation quantizer
        yq = getattr(layer, "yq")

        # insert the linear layer
        setattr(tmp, "linear", layer)

        sq = yq.scale.data / (tmp.linear.wq.scale.data * tmp.linear.aq.scale.data)
        
        # assign the scaling factor to the quantizer
        tmp.scaler.scale.data = sq
        tmp.scaler.zp.data = yq.zero_point.data

        # assign the dequantizer
        tmp.deq.scale.data = 1 / yq.scale.data
        
        # remove the original output quantizer
        tmp.linear.yq = nn.Identity()
        tmp.linear.aq = nn.Identity()
        return tmp
        

    def conv_bn_relu(self, cbr:List, l=-1.0, snxt:float=1.0, zpnxt:float=0.0, int_out:bool=False):
        assert len(cbr) == 3, "The input must include conv, bn, and relu modules"
        conv, bn, _ = cbr

        # fused layer
        tmp = QConvBNReLU(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, 
                        wbit=conv.wbit, abit=conv.abit, train_flag=conv.train_flag, int_out=int_out)
        
        # assign modules
        setattr(tmp, "conv", cbr[0])
        setattr(tmp, "bn", cbr[1])
        setattr(tmp, "relu", cbr[2])

        # quantization scalers
        sq = 1 / (tmp.conv.wq.scale.data * tmp.conv.aq.scale.data)
        
        # bn scaling
        std = torch.sqrt(bn.running_var.data + bn.eps)
        sbn = bn.weight.data.mul(sq) / std
        # bn bias
        bbn = bn.bias - bn.weight.mul(bn.running_mean.data).div(std)
        
        # scale and bias
        tmp.scaler.scale.data = sbn.mul(snxt)
        tmp.scaler.bias.data = bbn.mul(snxt).add(zpnxt)
        
        if isinstance(tmp.scaler, MulQuant):
            tmp.scaler.zp.data = zpnxt

        # replace batchnorm by the identity
        setattr(tmp, "bn", nn.Identity())

        # replace the activation quantizer by the Identity module
        if l > self.fpl-1:
            tmp.conv.aq = nn.Identity()
        
        return tmp
    
    def conv_relu(self, cr:List, l=-1.0, snxt:float=1.0, zpnxt:float=0.0, int_out:bool=False):
        assert len(cr) == 2, "The input must include conv and relu modules"

        conv, relu = cr
        
        # fused layer
        tmp = QConvReLU(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, 
                        wbit=conv.wbit, abit=conv.abit, train_flag=False, int_out=int_out)

        # quantization scalers
        sq = 1 / (conv.wq.scale.data * conv.aq.scale.data)
        
        # scaled bias
        sb = conv.bias.data.div(sq)
        conv.bias.data = sb

        # assign modules
        setattr(tmp, "conv", conv)
        setattr(tmp, "relu", relu)

        # next layer scaler
        tmp.scaler.scale.data = sq.mul(snxt)

        if isinstance(tmp.scaler, MulQuant):
            tmp.scaler.zp.data = zpnxt
        
        # replace the activation quantizer by the Identity module
        if l > self.fpl-1:
            tmp.conv.aq = nn.Identity()
        
        return tmp

    def fuse(self):
        """
        Fuse conv, bn, and relu layers
        """
        l = 0   # layer counter
        # initialize the model copy to avoid the mutated dict
        fused_model = self.model
        
        for name, module in self.model.named_children():
            if len(module) > 0:
                for n, m in module.named_children():
                    if isinstance(m, _QBaseConv2d):
                        # fetch the module
                        conv_bn_relu = self.groups[l]
                        bn = conv_bn_relu[1]

                        self.flag = True

                        if l < len(self.xscales)-1:
                            snxt = self.xscales[l+1]
                            int_out = True
                        else:
                            snxt = 1.0
                            if self.fpc:
                                int_out = False

                        # fused layer
                        tmp = QConvBNReLU(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, 
                                        wbit=m.wbit, abit=m.abit, train_flag=m.train_flag, int_out=int_out)

                        # assign modules
                        setattr(tmp, "conv", conv_bn_relu[0])
                        # setattr(tmp, "bn", conv_bn_relu[1])
                        setattr(tmp, "relu", conv_bn_relu[2])

                        # quantization scalers
                        sq = 1 / (tmp.conv.wq.scale.data * tmp.conv.aq.scale.data)

                        # bn scaling
                        std = torch.sqrt(bn.running_var.data + bn.eps)
                        sbn = bn.weight.data.mul(sq) / std
                        # bn bias
                        bbn = bn.bias - bn.weight.mul(bn.running_mean.data).div(std)
                        
                        # scale and bias
                        tmp.scaler.scale.data = sbn.mul(snxt)
                        tmp.scaler.bias.data = bbn.mul(snxt)

                        # replace batchnorm by the identity
                        setattr(tmp, "bn", nn.Identity())

                        # replace the activation quantizer by the Identity module
                        if l > self.fpl-1:
                            tmp.conv.aq = nn.Identity()

                        # update module
                        setattr(module, n, tmp)
                        
                        # increment
                        l += 1
                    elif isinstance(m, nn.BatchNorm2d) and self.flag:
                        tmp = nn.Identity()
                        
                        # replace bn by identity
                        setattr(module, n, tmp)
                    
                    elif isinstance(m, nn.ReLU) and self.flag:
                        tmp = nn.Identity()

                        # replace relu by identity
                        setattr(module, n, tmp)

                        # reset
                        self.flag = False
                        
                setattr(fused_model, name, module)
        
        return fused_model

