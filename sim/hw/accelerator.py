from typing import List
import numpy as np
import math
import torch
import torch.nn as nn

from hw.alu.alu_unit import PE
from model_profile.meters.layer_dim_profiler import LayerDim

class Accelerator:
    ### Global variable
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, 
                 pe: PE,
                 pe_array_dim: List[int],
                 model_name: str):
        self.model_name     = model_name
        self.pe             = pe
        self.pe_array_dim   = {'h': pe_array_dim[0], 'w': pe_array_dim[1]}
        self.total_pe_count = np.prod(pe_array_dim)

        self._init_model_profiler(model_name)
    
    def _init_model_profiler(self, model_name):
        dim_profiler = LayerDim(model_name)

        # format: {layer_name: [k, k, in_channel, out_channel], ...}
        self.weight_dim = dim_profiler.weight_dim 
        # format: {layer_name: [batch_size, width, height, in_channel], ...}
        self.input_dim  = dim_profiler.input_dim
        # format: {layer_name: [batch_size, width, height, out_channel], ...}
        self.output_dim = dim_profiler.output_dim

        self.layer_name_list = dim_profiler.layer_name_list
    
    def _get_quantized_model(self):
        model_q = {}
        base_path = '/home/yc2367/BitVert_DNN'
        model_config_path = f'{base_path}/Baseline_Int8/{self.model_name}'
        tensor_path = f'{model_config_path}/tensors'

        for name in self.layer_name_list:
            w_tensor_file = f'{tensor_path}/{name}.ops_x2.pt'
            w_tensor = torch.load(w_tensor_file)
            if len(w_tensor.shape) == 2: # transpose fully-connected layer
                w_tensor = w_tensor.permute(1, 0)
            model_q[name] = w_tensor
        return model_q

    def _init_mem(self):
        raise NotImplementedError('ERROR! No implementation of function _init_mem()')
