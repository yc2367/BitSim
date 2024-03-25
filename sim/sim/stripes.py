import math
import torch.nn as nn

from typing import List
from hw.mem.mem_instance import MemoryInstance
from hw.alu.alu_unit import BitSerialPE
from hw.alu.alu_array import BitSerialPEArray
from hw.accelerator import Accelerator

# Stripes accelerator
class Stripes(Accelerator):

    PE_GROUP_SIZE = 16 # length of the dot product inside one PE
    PE_ENERGY = 3
    PE_AREA = 1

    def __init__(self, 
                 input_precision_s: int, # bit-serial operand precision
                 input_precision_p: int, # bit-parallel operand precision
                 pe_array_dim: List[int],
                 model_name: str,
                 model: nn.Module): # model comes from "BitSim/sim.model_profile/models/models.py
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
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            print(name)
            if len(w_dim) == 4:
                cin = i_dim[3]
                cw  = w_dim[2]
                print(i_dim, w_dim, o_dim)
                if cin == cw:
                    self.cycle += self._calc_conv2d_cycle(w_dim, o_dim)
                else: # depthwise conv
                    self.cycle += self._calc_dwconv_cycle(w_dim, i_dim, o_dim)
            else:
                print(i_dim, w_dim, o_dim)
                self.cycle += self._calc_fc_cycle(w_dim, o_dim)
    
    def _calc_conv2d_cycle(self, w_dim, o_dim):
        pe_group_size = self.PE_GROUP_SIZE
        pe_row = self.pe_array_dim['h']
        pe_col = self.pe_array_dim['w']
        w_prec = self.pe.input_precision_s

        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, _ = o_dim

        # cycle_kernel:       number of cycles to process a kernel
        # cycle_out_channel:  number of cycles along output channel
        # cycle_out_width:    number of cycles along output width
        # cycle_out_height:   number of cycles along output height
        if cw < pe_group_size:
            cycle_kernel   = math.ceil(cw * (k**2) / pe_group_size)
        else:
            cycle_kernel   = math.ceil(cw / pe_group_size) * (k**2)
        cycle_out_channel  = math.ceil(cout / pe_row)
        cycle_out_width    = math.ceil(ow / pe_col)
        cycle_out_height   = oh

        cycle_per_batch = (cycle_kernel * cycle_out_channel * cycle_out_width * cycle_out_height) * w_prec
        total_cycle = cycle_per_batch * batch_size
        print(total_cycle)
        return total_cycle
    
    def _calc_dwconv_cycle(self, w_dim, i_dim, o_dim):
        pe_group_size = self.PE_GROUP_SIZE
        pe_col = self.pe_array_dim['w']
        w_prec = self.pe.input_precision_s

        # kernel size, kernel input channel, output channel
        _, _, _, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, _ = o_dim

        assert cin != cw, 'Not a depth-wise convolution!'

        # cycle_kernel:       number of cycles to process a kernel
        # cycle_out_channel:  number of cycles along output channel
        # cycle_out_width:    number of cycles along output width
        # cycle_out_height:   number of cycles along output height
        if cw < pe_group_size:
            cycle_kernel   = math.ceil(cw * (k**2) / pe_group_size)
        else:
            cycle_kernel   = math.ceil(cw / pe_group_size) * (k**2)
        cycle_out_channel  = cout
        cycle_out_width    = math.ceil(ow / pe_col)
        cycle_out_height   = oh

        cycle_per_batch = (cycle_kernel * cycle_out_channel * cycle_out_width * cycle_out_height) * w_prec
        total_cycle = cycle_per_batch * batch_size
        print(total_cycle)
        return total_cycle

    def _calc_fc_cycle(self, w_dim, o_dim):
        pe_group_size = self.PE_GROUP_SIZE
        pe_row = self.pe_array_dim['h']
        pe_col = self.pe_array_dim['w']
        w_prec = self.pe.input_precision_s

        # kernel size, input channel, output channel
        cin, cout = w_dim
        # batch size, output feature width, output channel
        batch_size, _ = o_dim

        # cycle_in_channel:   number of cycles along input channel
        # cycle_out_channel:  number of cycles along output channel
        cycle_in_channel  = math.ceil(cin / pe_group_size)
        cycle_out_channel = math.ceil(cout / pe_row)
        cycle_batch       = math.ceil(batch_size / pe_col)

        total_cycle = (cycle_in_channel * cycle_out_channel * cycle_batch) * w_prec
        print(total_cycle)
        return total_cycle

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
        


    

    