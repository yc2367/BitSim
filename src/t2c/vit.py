import copy
import torch.nn as nn

from src.t2c.convert import get_parent_name
from src.module.attention import QAttention
from src.module.fuse import LinearMulShift, MulQuant
from src.module.base import _QBaseLinear


class ViTFuser(object):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        self.q_layers = ["blocks.10.attn", "blocks.11.attn"]

    def inference(self):
        """
        Switch to inference mode
        """
        for n, m in self.model.named_modules():
            if hasattr(m, "inference"):
                m.inference()

    def layers(self):
        pass

    def fused_linear(self, linear:_QBaseLinear, int_out:bool=True, obit:int=32):
        f = LinearMulShift(linear.in_features, linear.out_features, 
                linear.wq.nbit, linear.aq.nbit, linear.train_flag, int_out=int_out, obit=obit)
        setattr(f, "linear", linear)
        return f

    def attn_fuser(self, module:QAttention):
        module.proj.wq.dequantize = False
        module.proj.aq.dequantize = False
        
        fproj = self.fused_linear(module.proj, int_out=False)
        sproj = 1 / (module.proj.wq.scale * module.proj.aq.scale)
        bias = fproj.linear.bias.data
        
        # update the scaler
        fproj.scaler.scale.data.copy_(sproj)
        fproj.scaler.bias = bias

        # remove the bias of the original layer
        fproj.linear.bias.data.fill_(0.0)

        setattr(module, "proj", fproj)
        return module
    
    def qkv_fuser(self, module:QAttention):
        module.xq.dequantize = False
        module.qqkv.dequantize = False
        module.qkv.wq.dequantize = False

        sqkv = module.qqkv.scale / (module.qkv.wq.scale * module.qkv.aq.scale)
        qbias = module.qkv.bias.mul(module.qqkv.scale)

        q = MulQuant(nbit=module.qqkv.nbit)
        q.scale.data.copy_(sqkv)
        q.bias.data = qbias

        # remove qqkv
        setattr(module, "qqkv", q)

        # update the attention scaling
        module.attn_scale.scale.data.mul_(module.qqkv.scale.pow(2))
        return module
    
    def fuse(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        for n, m in self.model.named_modules():
            if isinstance(m, QAttention):
                if n in self.q_layers:
                    parent_name, name = get_parent_name(n)
                    
                    # module = self.attn_fuser(m)
                    module = self.qkv_fuser(m)
                    setattr(modules[parent_name], name, module)
                
        return self.model
        