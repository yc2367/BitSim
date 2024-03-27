import torch
import torch.nn as nn
from software.util.bin_int_convert import int_to_signMagnitude, int_to_twosComplement
from sim.sim.model_quantized import MODEL

def pragmatic_calc_conv2d_cycle(model_name, layer_name, w_dim, o_dim, device='cuda'):
    model = MODEL[model_name].cpu()
    for n, m in model.named_modules():
        if ( layer_name == n ):
            w = m.weight()
            wq = torch.int_repr(w)
            wqb_signMagnitude = int_to_signMagnitude()
