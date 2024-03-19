"""
Learned Step Size Quantization
"""

import torch
from src.module.base import _QBase

def lp_loss(pred, target, p=2.0, reduction='none'):
    """
    loss function measured in lp norm
    """
    if reduction == 'none':
        return (pred-target).abs().pow(p).sum(1).mean()
    else:
        return (pred-target).abs().pow(p).mean()
    
def round_ste(x:torch.Tensor):
    """
    Quantization with STE
    """
    return (x.round() - x).detach() + x


class LSQ(_QBase):
    def __init__(self, nbit: int = 8, 
                 train_flag: bool = True, unsigned: bool = True):
        super().__init__(nbit, train_flag, unsigned)
        self.train_flag = train_flag
        self.unsigned = unsigned

        # register learnable parameter 
        self.register_parameter("delta", torch.nn.Parameter(torch.tensor(1.0)))
        
        # initialization flag
        self.initialize = False

        self.prev_int_frame = None
        self.round_err = 0.0
        self.frame_err_all = []

        # upper and lower bound
        if not self.unsigned:
            self.qlb = -2**(self.nbit-1)
            self.qub = 2**(self.nbit-1) - 1
        else:
            self.qlb = 0
            self.qub = 2**(self.nbit) - 1
    
    def get_fp_range(self, x:torch.Tensor):
        y = torch.flatten(x, start_dim=1)
        batch_min = torch.min(y, 1)[0].mean()
        batch_max = torch.max(y, 1)[0].mean()
        return batch_min, batch_max
    
    def quantize(self, x:torch.Tensor, xmin, xmax):
        delta = (xmax - xmin) / (2 ** self.nbit - 1)
        zero_point = (-xmin / delta).round()

        xint = torch.round(x / delta)
        xq = torch.clamp(xint + zero_point, self.qlb, self.qub)
        xdq = (xq - zero_point) * delta
        return xdq
    
    def initialize_qparam(self, x:torch.Tensor):
        """
        Find the optimal scaling factor in the first batch
        """
        x_min, x_max = self.get_fp_range(x)
        best_loss = 1e+10

        for i in range(80):
            new_max = x_max * (1.0 - (i * 0.01))
            new_min = x_min * (1.0 - (i * 0.01))

            # quantize and dequantize for mse 
            xdq = self.quantize(x, new_min, new_max)
            loss = lp_loss(xdq, x, p=2.4, reduction='all')

            if loss < best_loss:
                best_loss = loss
                delta = (new_max - new_min) / (self.qub - self.qlb)
                zero_point = (-new_min / delta).round()
        
        return delta, zero_point

    def q(self, x:torch.Tensor):
        if not self.initialize:
            if self.train_flag:
                delta, zero_point = self.initialize_qparam(x)
                self.delta.data = delta
                self.zero_point.data = zero_point
                
                self.initialize = True

        # quantize
        xr = round_ste(x / self.delta) + self.zero_point
        xq = torch.clamp(xr, min=self.qlb, max=self.qub)

        # dequantize
        xdq = (xq - self.zero_point) * self.delta

        return xdq
    
    def trainFunc(self, input: torch.Tensor):
        xdq = self.q(input)
        # update the buffer
        self.scale.data.copy_(1 / self.delta.data)
        return xdq
    
    def evalFunc(self, input: torch.Tensor):
        xr = round_ste(input * self.scale) + self.zero_point
        xq = torch.clamp(xr, min=self.qlb, max=self.qub)
        xdq = (xq - self.zero_point)

        if self.dequantize:
            xdq = xdq.div(self.scale)

        return xdq
    
    def extra_repr(self) -> str:
        return super().extra_repr() + f", delta={self.delta.data.item():.2e}"
