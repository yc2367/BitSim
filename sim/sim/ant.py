import math
import torch
import torch.nn as nn

from typing import List
from hw.mem.mem_instance import MemoryInstance
from sim.stripes import Stripes

# Pragmatic accelerator
class Ant(Stripes):
    PR_SCALING = 1.5 # scaling factor to account for post placement and routing
    PRECISION_SCALING = 0.8 # scaling factor to account for 6-bit ANT
    
    DISPATCHER_ENERGY_PER_COL = 0.072625 
    PE_ENERGY = 0.28125 * PR_SCALING * PRECISION_SCALING # energy per PE
    W_REG_ENERGY_PER_ROW = 0.46 * PR_SCALING # energy (pJ) of the weight scheduler for a PE row
    I_REG_ENERGY_PER_COL = (0.53 + DISPATCHER_ENERGY_PER_COL) * PR_SCALING # energy (pJ) of the activation register file for a PE column
    PE_AREA = 1

    def __init__(self, 
                 input_precision_s: int, # bit-serial operand precision
                 input_precision_p: int, # bit-parallel operand precision
                 pe_dotprod_size: int, # length of the dot product inside one PE
                 pe_array_dim: List[int],
                 model_name: str):
        super().__init__(input_precision_s, input_precision_p, pe_dotprod_size, 
                         pe_array_dim, model_name)    
        self.model_q = self._get_quantized_model() # quantized model
    
    def _check_layer_mem_size(self):
        self._w_mem_required = {}
        self._i_mem_required = {}
        self._o_mem_required = {}   
        w_prec = self.pe.input_precision_p
        i_prec = self.pe.input_precision_p
        for layer_idx, name in enumerate(self.layer_name_list):
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    # kernel size, kernel input channel, output channel
                    k, _, cw, cout = w_dim
                    # batch size, input feature width, input feature height, input channel
                    batch_size, iw, ih, cin = i_dim
                    # batch size, output feature height, output feature width, output channel
                    _, oh ,ow, _ = o_dim
                    
                    self._w_mem_required[name] = math.ceil(cw * w_prec / 8) * k**2 * cout
                    self._i_mem_required[name] = math.ceil(cin * i_prec / 8) * ih * iw 
                    self._o_mem_required[name] = math.ceil(cout * i_prec / 8) * oh * ow 
                else:
                    # input channel, output channel
                    cin, cout = w_dim
                    # batch size, sample size, output channel
                    batch_size, token_num, _ = o_dim

                    self._w_mem_required[name] = math.ceil(cin * w_prec / 8) * cout
                    self._i_mem_required[name] = math.ceil(cin * i_prec / 8)  * token_num
                    if layer_idx == (len(self.layer_name_list) - 1):
                        self._o_mem_required[name] = 0
                    else:
                        self._o_mem_required[name] = math.ceil(cout * i_prec / 8)  * token_num

    def _init_mem(self):
        w_prec = 6
        w_sram_bank = self.pe_array_dim['h'] * w_prec / 8 
        w_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 240 * 1024*8, 
                            'bank_count': w_sram_bank, 
                            'rw_bw': (self.pe_dotprod_size * 8) * w_sram_bank, 
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.w_sram = MemoryInstance('w_sram', w_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=None, min_w_granularity=64,  
                                     get_cost_from_cacti=True, double_buffering_support=False)
        
        i_prec = 6
        i_sram_bank = self.pe_array_dim['w'] * i_prec / 8 
        i_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 270 * 1024*8, 
                            'bank_count': i_sram_bank, 
                            'rw_bw': (self.pe_dotprod_size * 8) * i_sram_bank,
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.i_sram = MemoryInstance('i_sram', i_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=64, min_w_granularity=64, 
                                     get_cost_from_cacti=True, double_buffering_support=False)
        
        dram_config = {
                        'technology': 0.028,
                        'mem_type': 'dram', 
                        'size': 1e9 * 8, 
                        'bank_count': 1, 
                        'rw_bw': 64,
                        'r_port': 0, 
                        'w_port': 0, 
                        'rw_port': 1,
                    }
        self.dram = MemoryInstance('dram', dram_config, 
                                    r_cost=1000, w_cost=1000, latency=1, area=0, 
                                    min_r_granularity=64, min_w_granularity=64, 
                                    get_cost_from_cacti=False, double_buffering_support=False)
     