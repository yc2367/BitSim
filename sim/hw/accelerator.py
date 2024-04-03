from typing import List
import numpy as np
import torch
import torch.nn as nn

from hw.alu.alu_unit import PE
from model_profile.meters.dim import DIM

class Accelerator:
    ### Global variable
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, 
                 pe: PE,
                 pe_array_dim: List[int],
                 model_name: str,
                 model: nn.Module):
        self.pe             = pe
        self.pe_array_dim   = {'h': pe_array_dim[0], 'w': pe_array_dim[1]}
        self.total_pe_count = np.prod(pe_array_dim)

        self._init_model_profiler(model_name, model)
    
    def _init_model_profiler(self, model_name, model):
        dim_profiler = DIM(model_name, model, device=self.DEVICE, input_size=224)
        dim_profiler.fit()

        # format: {layer_name: [k, k, in_channel, out_channel], ...}
        self.weight_dim = dim_profiler.weight_dim 
        # format: {layer_name: [batch_size, width, height, in_channel], ...}
        self.input_dim  = dim_profiler.input_dim
        # format: {layer_name: [batch_size, width, height, out_channel], ...}
        self.output_dim = dim_profiler.output_dim

        self.layer_name_list = dim_profiler.layer_name_list
    
    def _init_mem(self):
        raise NotImplementedError
    
    def _check_layer_mem_size(self):
        raise NotImplementedError
    
    def _calc_num_mem_refetch(self):
        raise NotImplementedError



    
    
    