import math
import torch.nn as nn
import numpy as np

from typing import List
from hw.mem.mem_instance import MemoryInstance
from hw.alu.alu_unit import BitSerialPE
from hw.accelerator import Accelerator

# Stripes accelerator
class Stripes(Accelerator):
    PR_SCALING = 1.5 # scaling factor to account for post placement and routing
    DISPATCHER_ENERGY_PER_COL = 0.072625
    PE_ENERGY = 0.28125 * PR_SCALING # energy per 8-way DP PE
    #PE_ENERGY = 0.23375 * PR_SCALING
    W_REG_ENERGY_PER_ROW = 0.46 * PR_SCALING # energy (pJ) of the weight scheduler for a PE row
    #W_REG_ENERGY_PER_ROW = 0.23 * PR_SCALING
    I_REG_ENERGY_PER_COL = (0.53 + DISPATCHER_ENERGY_PER_COL) * PR_SCALING # energy (pJ) of the activation register file for a PE column
    #I_REG_ENERGY_PER_COL = (0.2625 + DISPATCHER_ENERGY_PER_COL) * PR_SCALING 
    PE_AREA = 1

    def __init__(self, 
                 input_precision_s: int, # bit-serial operand precision
                 input_precision_p: int, # bit-parallel operand precision
                 pe_dotprod_size: int, # length of the dot product inside one PE
                 pe_array_dim: List[int],
                 model_name: str,
                 model: nn.Module, # model comes from "BitSim/sim.model_profile/models/models.py
                 init_mem: bool=True): 
        assert len(pe_array_dim) == 2, \
            f'PE array must have 2 dimensions, but you gave {len(pe_array_dim)}'
        
        self.pe_dotprod_size = pe_dotprod_size
        pe = BitSerialPE(input_precision_s, input_precision_p, 
                         pe_dotprod_size, self.PE_ENERGY, self.PE_AREA)
        super().__init__(pe, pe_array_dim, model_name, model)

        self.cycle_compute = None
        if init_mem:
            self._init_mem()
            self._check_layer_mem_size()
            self._calc_num_mem_refetch()
    
    def _calc_eff_ops(self):
        num_eff_ops = 0
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    num_eff_ops += self._calc_eff_ops_conv(w_dim, o_dim)
                else:
                    num_eff_ops += self._calc_eff_ops_fc(w_dim, o_dim)
        self.total_eff_ops = num_eff_ops
    
    def _calc_eff_ops_conv(self, w_dim, o_dim):
        w_prec = self.pe.input_precision_s
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature height, output feature width, output channel
        batch_size, oh , ow, _ = o_dim
        num_eff_ops = (k**2) * cw * cout * oh * ow 
        return num_eff_ops * w_prec / self.pe_dotprod_size
    
    def _calc_eff_ops_fc(self, w_dim, o_dim):
        w_prec = self.pe.input_precision_s
        # input channel, output channel
        cin, cout = w_dim
        # batch size, sample size, output channel
        batch_size, token_num, _ = o_dim
        num_eff_ops = cin * cout * token_num 
        return num_eff_ops * w_prec / self.pe_dotprod_size

    def calc_cycle(self):
        self._calc_compute_cycle()
        self._calc_dram_cycle() 
        total_cycle = 0
        total_cycle_compute = 0
        for name in self.layer_name_list:
            cycle_layer_compute = self._layer_cycle_compute[name]
            cycle_layer_dram    = self._layer_cycle_dram[name]
            total_cycle_compute += cycle_layer_compute
            total_cycle += max(cycle_layer_compute, cycle_layer_dram)
        self.cycle_compute = total_cycle_compute
        return total_cycle_compute, total_cycle
    
    def _calc_compute_cycle(self):
        self._layer_cycle_compute = {}
        w_prec = self.pe.input_precision_s
        self._read_layer_input = True
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            tile_layer = 0
            cycle_layer_compute = 0
            if w_dim is not None:
                if len(w_dim) == 4:
                    cin = i_dim[3]
                    cw  = w_dim[2]
                    if cin == cw: 
                        tile_layer = self._calc_tile_conv2d(w_dim, o_dim)
                    else: # depthwise conv
                        tile_layer = self._calc_tile_dwconv(w_dim, i_dim, o_dim)
                else:
                    tile_layer = self._calc_tile_fc(w_dim, o_dim)
                cycle_layer_compute = tile_layer * w_prec
                self._layer_cycle_compute[name] = cycle_layer_compute
            '''
            else:
                self._layer_cycle_compute[name] = 0
            '''
    
    def calc_pe_array_tile(self):
        total_tile = 0
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    cin = i_dim[3]
                    cw  = w_dim[2]
                    if cin == cw: 
                        total_tile += self._calc_tile_conv2d(w_dim, o_dim)
                    else: # depthwise conv
                        total_tile += self._calc_tile_dwconv(w_dim, i_dim, o_dim)
                else:
                    total_tile += self._calc_tile_fc(w_dim, o_dim)
        return total_tile
    
    def _calc_tile_conv2d(self, w_dim, o_dim):
        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature height, output feature width, output channel
        batch_size, oh ,ow, _ = o_dim

        # tile_kernel: number of tiles to process a kernel
        # tile_cout:   number of tiles along output channel
        # tile_ow:     number of tiles along output width
        # tile_oh:     number of tiles along output height
        if ( k**2 > cw ):
            tile_kernel   = math.ceil((k**2) / pe_group_size) * cw
        else:
            tile_kernel   = math.ceil(cw / pe_group_size) * (k**2)
        tile_cout  = math.ceil(cout / num_pe_row)
        tile_ow    = math.ceil(ow / num_pe_col)
        tile_oh    = oh

        tile_per_batch = (tile_kernel * tile_cout * tile_ow * tile_oh)
        total_tile = tile_per_batch 
        return total_tile
    
    def _calc_tile_dwconv(self, w_dim, i_dim, o_dim):
        pe_group_size = self.pe_dotprod_size
        num_pe_col = self.pe_array_dim['w']

        # kernel size, kernel input channel, output channel
        _, _, _, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature height, output feature width, output channel
        batch_size, oh ,ow, _ = o_dim

        assert cin != cw, 'Not a depth-wise convolution!'

        # tile_kernel: number of tiles to process a kernel
        # tile_cout:   number of tiles along output channel
        # tile_ow:     number of tiles along output width
        # tile_oh:     number of tiles along output height
        tile_kernel = math.ceil((k**2) / pe_group_size) * cw
        tile_cout   = cout
        tile_ow     = math.ceil(ow / num_pe_col)
        tile_oh     = oh

        tile_per_batch = (tile_kernel * tile_cout * tile_ow * tile_oh)
        total_tile = tile_per_batch 
        return total_tile

    def _calc_tile_fc(self, w_dim, o_dim):
        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # input channel, output channel
        cin, cout = w_dim
        # batch size, sample size, output channel
        batch_size, token_num, _ = o_dim

        # tile_in_channel:   number of tiles along input channel
        # tile_cout:  number of tiles along output channel
        tile_in_channel  = math.ceil(cin / pe_group_size)
        tile_cout        = math.ceil(cout / num_pe_row)
        tile_batch       = math.ceil(batch_size * token_num / num_pe_col)

        total_tile = (tile_in_channel * tile_cout * tile_batch)
        return total_tile
    
    def _calc_dram_cycle(self):
        self._layer_cycle_dram = {}
        i_prec = self.pe.input_precision_p
        w_prec_pad = self.pe.input_precision_p
        dram_bandwidth = self.dram.rw_bw * 2 # DDR
        size_sram_i = self.i_sram.size / 8
        self._read_layer_input = True

        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            num_dram_fetch_w, num_dram_fetch_i = self._layer_mem_refetch[name]
            cycle_layer_dram = 0
            if w_dim is not None:
                cycle_dram_load_w = self._w_mem_required[name] * w_prec_pad / dram_bandwidth
                cycle_dram_load_w *= num_dram_fetch_w

                i_mem_required = self._i_mem_required[name]
                if self._read_layer_input:
                    cycle_dram_load_i = i_mem_required * i_prec / dram_bandwidth 
                else:
                    cycle_dram_load_i = 0
                cycle_dram_load_i *= num_dram_fetch_i

                o_mem_required = self._o_mem_required[name]
                if ( i_mem_required + o_mem_required ) < size_sram_i:
                    cycle_dram_write_o = 0
                    self._read_layer_input = False
                else:
                    cycle_dram_write_o = o_mem_required * i_prec / dram_bandwidth
                    self._read_layer_input = True
                
                cycle_layer_dram = cycle_dram_load_w + cycle_dram_write_o + cycle_dram_load_i
            self._layer_cycle_dram[name] = int(cycle_layer_dram)
    
    def calc_compute_energy(self, use_old=True):
        if not use_old:
            compute_energy = self.total_eff_ops * self.PE_ENERGY
        else:
            num_pe = self.total_pe_count
            if self.cycle_compute is None:
                self.cycle_compute, _ = self.calc_cycle()
            num_cycle_compute = self.cycle_compute
            compute_energy = self.PE_ENERGY * num_pe * num_cycle_compute
        return compute_energy
    
    def calc_sram_rd_energy(self):
        w_sram_rd_cost = self.w_sram.r_cost
        i_sram_rd_cost = self.i_sram.r_cost
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']
        if self.cycle_compute is None:
            self.cycle_compute, _ = self.calc_cycle()
        num_cycle_compute = self.cycle_compute
        num_tile = self.calc_pe_array_tile()

        sram_rd_energy = num_tile * (w_sram_rd_cost + i_sram_rd_cost)
        w_reg_energy = self.W_REG_ENERGY_PER_ROW * num_pe_row * num_cycle_compute
        # The activation register is accessed every tile
        i_reg_energy = self.I_REG_ENERGY_PER_COL * num_pe_col * num_tile
        total_energy = sram_rd_energy + w_reg_energy + i_reg_energy
        return total_energy
    
    def calc_sram_wr_energy(self):
        total_energy = 0
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            w_prec = self.pe.input_precision_p
            i_prec = self.pe.input_precision_p
            if w_dim is not None:
                if len(w_dim) == 4:
                    total_energy += self._calc_sram_wr_energy_conv(name, w_dim, i_dim, o_dim, 
                                                                   w_prec, i_prec)
                else:
                    total_energy += self._calc_sram_wr_energy_fc(name, w_dim, i_dim, o_dim,
                                                                 w_prec, i_prec)
        return total_energy
    
    def _calc_sram_wr_energy_conv(self, layer_name, w_dim, i_dim, o_dim, w_prec, i_prec):
        w_sram_wr_cost = self.w_sram.w_cost_min
        i_sram_wr_cost = self.i_sram.w_cost_min
        w_sram_min_wr_bw = self.w_sram.w_bw_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        # kernel size, kernel input channel, output channel
        batch_size, iw, ih, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature height, output feature width, output channel
        _, oh ,ow, _ = o_dim
        
        # write energy, read from DRAM and write to SRAM
        num_w_sram_wr = math.ceil(cw * w_prec / w_sram_min_wr_bw) * (k**2) * cout
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * num_fetch_w

        num_i_sram_wr  = math.ceil(cin * i_prec / i_sram_min_wr_bw) * ih * iw 
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i

        num_o_sram_wr  = math.ceil(cout * i_prec / i_sram_min_wr_bw) * oh * ow 
        energy_o_sram_wr = num_o_sram_wr * i_sram_wr_cost

        total_energy = energy_w_sram_wr + energy_i_sram_wr + energy_o_sram_wr
        return total_energy
    
    def _calc_sram_wr_energy_fc(self, layer_name, w_dim, i_dim, o_dim, w_prec, i_prec):
        w_sram_wr_cost = self.w_sram.w_cost_min
        i_sram_wr_cost = self.i_sram.w_cost_min
        w_sram_min_wr_bw = self.w_sram.w_bw_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        # input channel, output channel
        cin, cout = w_dim
        # batch size, sample size, input channel
        batch_size, _, _ = i_dim
        # batch size, sample size, output channel
        _, token_num, _ = o_dim

        # write energy, read from DRAM and write to SRAM
        num_w_sram_wr = math.ceil(cin * w_prec / w_sram_min_wr_bw) * cout
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * num_fetch_w

        num_i_sram_wr  = math.ceil(cin * i_prec / i_sram_min_wr_bw)  * token_num
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i

        num_o_sram_wr  = math.ceil(cout * i_prec / i_sram_min_wr_bw)  * token_num
        if token_num == 1: # CNN last FC layer
            energy_o_sram_wr = 0
        else:
            energy_o_sram_wr = num_o_sram_wr * i_sram_wr_cost

        total_energy = energy_w_sram_wr + energy_i_sram_wr + energy_o_sram_wr
        return total_energy
    
    def calc_dram_energy(self):
        energy = 0
        self._read_layer_input = True
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    energy += self._calc_dram_energy_conv(name)
                else:
                    energy += self._calc_dram_energy_fc(name)
        return energy

    def _calc_dram_energy_conv(self, layer_name):
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost
        wr_cost = self.dram.w_cost
        size_sram_i = self.i_sram.size / 8
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]
        
        # energy_input:  energy to read input feature map
        # energy_output: energy to write output feature map
        # energy_weight: energy to read weight
        w_mem_required = self._w_mem_required[layer_name]
        energy_weight = w_mem_required * 8 / bus_width * rd_cost

        i_mem_required = self._i_mem_required[layer_name]
        if self._read_layer_input:
            energy_input = i_mem_required * 8 / bus_width * rd_cost 
        else:
            energy_input = 0

        o_mem_required = self._o_mem_required[layer_name]
        if ( i_mem_required + o_mem_required ) < size_sram_i:
            energy_output = 0
            self._read_layer_input = False
        else:
            energy_output = o_mem_required * 8 / bus_width * wr_cost
            self._read_layer_input = True
        
        energy_weight *= num_fetch_w
        energy_input  *= num_fetch_i
        total_energy = energy_weight + energy_input + energy_output
        return total_energy
    
    def _calc_dram_energy_fc(self, layer_name):
        size_sram_i = self.i_sram.size / 8
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost
        wr_cost = self.dram.w_cost
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        # energy_weight: energy to read weight from DRAM
        w_mem_required = self._w_mem_required[layer_name]
        energy_weight = w_mem_required * 8 / bus_width * rd_cost
        
        # energy_input:  energy to read input feature map from DRAM
        i_mem_required = self._i_mem_required[layer_name]
        if self._read_layer_input:
            energy_input  = i_mem_required * 8 / bus_width * rd_cost
        else:
            energy_input = 0

        o_mem_required = self._o_mem_required[layer_name]
        # energy_output: energy to write output feature map
        # if the buffer is enough to hold the output feature map, no need to write to DRAM
        if ( i_mem_required + o_mem_required ) < size_sram_i:
            energy_output = 0
            self._read_layer_input = False
        else:
            if o_mem_required == 0: # last layer, no need to write output feature
                energy_output = 0
                self._read_layer_input = True
            else:
                energy_output = o_mem_required * 8 / bus_width * wr_cost
                self._read_layer_input = True

        energy_weight *= num_fetch_w
        energy_input  *= num_fetch_i
        total_energy = energy_weight + energy_input + energy_output
        return total_energy
    
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

    def _calc_num_mem_refetch(self):
        # If the on-chip buffer size is not big enough, 
        # we need to refetch input tiles or weight tiles from DRAM
        self._layer_mem_refetch = {}
        size_sram_w   = self.w_sram.size / 8
        size_sram_i   = self.i_sram.size / 8
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            if w_dim is not None:
                w_mem_required = self._w_mem_required[name]
                i_mem_required = self._i_mem_required[name]
                if ( w_mem_required > size_sram_w ) and ( i_mem_required > size_sram_i ):
                    # need DRAM refetch
                    num_refetch_input  = math.ceil(w_mem_required / size_sram_w)
                    num_refetch_weight = math.ceil(i_mem_required / size_sram_i)
                    total_fetch_weight = num_refetch_weight * w_mem_required
                    total_fetch_input  = num_refetch_input * i_mem_required
                    #print('Need DRAM refetch ...')
                    #print(f'w_dim: {w_dim}, i_dim: {i_dim}')
                    if ( total_fetch_weight + i_mem_required ) < ( total_fetch_input + w_mem_required ):
                        #print(f'Refetch weight for {num_refetch_weight} times ...')
                        # refetch all weight for every input tile
                        self._layer_mem_refetch[name] = (num_refetch_weight, 1)
                    else:
                        #print(f'Refetch input for {num_refetch_input} times ...')
                        # refetch all input for every weight tile
                        self._layer_mem_refetch[name] = (1, num_refetch_input)
                else:
                    # no need refetch
                    self._layer_mem_refetch[name] = (1, 1)

    def _init_mem(self):
        w_prec = self.pe.input_precision_p
        w_sram_bank = self.pe_array_dim['h'] # one bank feeds 1 PE row
        w_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 288 * 1024*8, 
                            'bank_count': w_sram_bank, 
                            'rw_bw': (self.pe_dotprod_size * w_prec) * w_sram_bank, 
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.w_sram = MemoryInstance('w_sram', w_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=None, 
                                     min_w_granularity=64, 
                                     get_cost_from_cacti=True, double_buffering_support=False)
        
        i_prec = self.pe.input_precision_p
        i_sram_bank = self.pe_array_dim['w'] # one bank feeds 1 PE columns
        i_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 256 * 1024*8, 
                            'bank_count': i_sram_bank, 
                            'rw_bw': (self.pe_dotprod_size * i_prec) * i_sram_bank,
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.i_sram = MemoryInstance('i_sram', i_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=64, 
                                     min_w_granularity=64, 
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
