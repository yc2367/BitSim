import torch.nn as nn

from typing import List
from hw.mem.mem_instance import MemoryInstance
from hw.alu.alu_unit import BitSerialPE
from hw.alu.alu_array import BitSerialPEArray
from hw.accelerator import Accelerator
from model_profile.meters.dim import DIM

# Stripes accelerator
class Stripes(Accelerator):

    PE_GROUP_SIZE = 16
    PE_ENERGY = 3
    PE_AREA = 1

    def __init__(self, 
                 input_precision_s: int, # bit-serial operand precision
                 input_precision_p: int, # bit-parallel operand precision
                 pe_array_dim: List[int],
                 model_name: str,
                 model: nn.Module):
        assert len(pe_array_dim) == 2, \
            f'PE array must have 2 dimensions, but you gave {len(pe_array_dim)}'
        
        self.pe_array_dim = {'h': pe_array_dim[0], 'w': pe_array_dim[1]}
        self.sram_energy = 0
        self.dram_energy = 0

        pe = BitSerialPE(input_precision_s, input_precision_p, 
                         self.PE_GROUP_SIZE, self.PE_ENERGY, self.PE_AREA)
        super().__init__(pe, self.pe_array_dim, model_name, model)
        self._init_mem()
    
    def calc_cycle(self):
        for name in self.layer_name_list:
            o_dim = self.output_dim[name]
            w_dim = self.weight_dim[name]
            if 'conv' in name:
                self.cycle += self._calc_conv_cycle(w_dim, o_dim)
            else
    
    def _calc_conv_cycle(self):

    
    def _init_mem(self):
        w_sram_bank = 16 # one bank feeds 2 PE rows
        w_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 9 * w_sram_bank * 1024*8, 
                            'bank_count': w_sram_bank, 
                            'rw_bw': (self.pe_array_dim['h'] * 8) * w_sram_bank, 
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.w_sram = MemoryInstance('w_sram', w_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=None, min_w_granularity=128, 
                                     get_cost_from_cacti=True, double_buffering_support=False)
        
        i_sram_bank = 16 # one bank feeds 2 PE columns
        i_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 8 * 1024*8 * i_sram_bank, 
                            'bank_count': i_sram_bank, 
                            'rw_bw': (self.pe_array_dim['w'] // 2 * 8) * i_sram_bank,
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.i_sram = MemoryInstance('i_sram', i_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=64, min_w_granularity=128, 
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
                                    r_cost=545, w_cost=560, latency=1, area=0, 
                                    min_r_granularity=64, min_w_granularity=64, 
                                    get_cost_from_cacti=False, double_buffering_support=False)
        
        print(self.dram.r_cost, self.dram.w_cost)


    

    