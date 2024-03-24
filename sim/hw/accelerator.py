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
        #self.model_w, self.model_i, self.model_o = self._init_model_profiler(model_name, model)
    
    def _init_model_profiler(self, model_name, model):
        profiler = DIM(model_name, model, device=self.DEVICE, input_size=224)
        profiler.fit()
        return profiler.w_dict, profiler.i_dict, profiler.o_dict
    
    def init_mem(self):
        raise NotImplementedError
    
    def get_pe_array_dim(self):
        return self.pe_array.dimension
    