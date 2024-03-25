from typing import Dict
import torch
import torch.nn as nn

from hw.alu.alu_unit import PE
from hw.alu.alu_array import PEArray
from model_profile.meters.dim import DIM

class Accelerator:
    ### Global variable
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, 
                 pe: PE,
                 pe_array_dim: Dict[str, int],
                 model_name: str,
                 model: nn.Module):
        self.pe            = pe
        self.pe_array_dim  = pe_array_dim
        self.pe_array      = PEArray(pe, pe_array_dim)
        (self.weight_dim, 
         self.input_dim, 
         self.output_dim,
         self.layer_name_list) = self._init_model_profiler(model_name, model)
        
        # number of cycles for inference
        self.cycle = 0
        # number of memory access
        self.num_mem_access = {'w_sram': 0, 'i_sram': 0, 'dram': 0}
    
    def _init_model_profiler(self, model_name, model):
        dim_profiler = DIM(model_name, model, device=self.DEVICE, input_size=224)
        dim_profiler.fit()
        return (dim_profiler.weight_dim, 
                dim_profiler.input_dim, 
                dim_profiler.output_dim, 
                dim_profiler.layer_name_list)
    
    def _init_mem(self):
        raise NotImplementedError
    
    def calc_cycle(self):
        raise NotImplementedError
    
    def get_pe_array_dim(self):
        return self.pe_array.dimension
    