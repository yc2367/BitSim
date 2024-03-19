import numpy as np 
import torch
import copy
import torch.nn as nn

from src.module.fuse import QConvReLU, QConvBNReLU, MulShift, ConvReLU
from src.module.base import _QBaseLinear
from src.module.convert import get_parent_name
from src.t2c.resnet import ResNet18Fuser, ResNet50Fuser
from src.t2c.vit import ViTFuser

from fxpmath import Fxp

FUSERS = {
    "resnet18": ResNet18Fuser,
    "resnet50": ResNet50Fuser,
    "vit_base": ViTFuser
}

class T2C(object):
    def __init__(self, model:nn.Module, swl:int, sfl:int, args):
        self.swl = swl
        self.sfl = sfl
        self.args = args

        self.swl = swl
        self.sfl = sfl
        self.args = args

        # model fusion
        fuser = FUSERS[str(args.model)](model)
        
        # get layer info
        fuser.layers()

        # switch to inference mode
        fuser.inference()
        
        # fuse layers
        fused_model = fuser.fuse()
        self.model = fused_model

    @property
    def sparsity(self):
        return self.compute_sparsity()

    def compute_sparsity(self):
        total_param = 0
        zeros = 0
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                w = m.weight
                total_param += w.numel()
                zeros += w[w.eq(0.)].numel()

        return zeros / total_param
    
    def f2fxp(self, val):
        vfix = Fxp(val, signed=True, n_word=self.swl, n_frac=self.sfl)
        vfix = vfix.base_repr(10)
        vnp = np.array(vfix).astype(float)
        return torch.from_numpy(vnp).cuda()

    def scale_bias2int(self, model:nn.Module):
        """
        Convert the pre-computed scaling factor and bias to high precision integer
        """
        qnn = copy.deepcopy(model)
        for n, m in qnn.named_modules():
            if isinstance(m, MulShift):
                m.fl = self.sfl
                scale = m.scale.cpu().numpy()
                bias = m.bias.cpu().numpy()

                # to numpy
                sint = self.f2fxp(scale)
                bint = self.f2fxp(bias)
                
                # insert back
                m.scale = sint.float()
                m.bias = bint.float()
        return qnn
    
    def fused_model(self):
        return self.model
    
    def nn2chip(self):
        """
        Insert the integer parameters back to the vanilla modules
        """

        modules = dict(self.model.named_modules(remove_duplicate=False))
        for n, m in modules.items():
            if isinstance(m, (QConvBNReLU, QConvReLU)):
                m.conv.weight.data.copy_(m.conv.qweight.data)
                m.conv.wq = nn.Identity()
            
            elif isinstance(m, _QBaseLinear):
                m.weight.data.copy_(m.qweight.data)
                m.wq = nn.Identity()
                
        return self.model
