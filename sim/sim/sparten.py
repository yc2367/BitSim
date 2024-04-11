import math, time
import torch
import torch.nn as nn
import numpy as np

from typing import List

from hw.accelerator import Accelerator
from hw.alu.alu_unit import PE
from hw.mem.mem_instance import MemoryInstance
from model_profile.meters.sparten_profiler import SpartenProfiler

# Pragmatic accelerator
class Sparten(Accelerator):
    LOCAL_OUTPUT_BUFFER_SIZE = 32
    PE_ENERGY = 1.3 # energy per PE, including MAC, prefix-sum, priority encoder
    PE_AREA = 1

    def __init__(self, 
                 input_precision: int, # operand precision
                 pe_dotprod_size: int, # length of the dot product inside one PE
                 pe_array_dim: List[int],
                 model_name: str,
                 model: nn.Module, # model comes from "BitSim/sim.model_profile/models/models.py
                 args):
        assert len(pe_array_dim) == 2, \
            f'PE array must have 2 dimensions, but you gave {len(pe_array_dim)}'
        self.input_precision = input_precision
        self.pe_dotprod_size = pe_dotprod_size
        (self.i_num_zero, 
         self.o_num_zero, 
         self.w_num_zero, 
         self.num_eff_ops) = self._get_sparse_info(model_name, model, pe_dotprod_size, args)
        
        pe = PE([input_precision, input_precision], 
                pe_dotprod_size, self.PE_ENERGY, self.PE_AREA)
        super().__init__(pe, pe_array_dim, model_name, model)
        self._calc_eff_ops()

        self._init_mem()
        self._check_layer_mem_size()
        self._calc_num_mem_refetch()
        self.cycle_compute = None
    
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
        self.cycle_dwconv = 0
        for name in self.layer_name_list:
            cycle_layer_compute = 0
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    cin = i_dim[3]
                    cw  = w_dim[2]
                    if cin == cw: 
                        cycle_layer_compute = self._calc_cycle_conv2d(name)
                    else: # depthwise conv
                        cycle_layer_compute = self._calc_cycle_dwconv(name)
                        self.cycle_dwconv += cycle_layer_compute
                else:
                    cycle_layer_compute = self._calc_cycle_fc(name)
                self._layer_cycle_compute[name] = cycle_layer_compute
            '''
            else:
                self._layer_cycle_compute[name] = 0
            '''

    def _calc_cycle_conv2d(self, layer_name):
        total_cycle = 0
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']
        num_eff_ops = self.num_eff_ops[layer_name] # dimension: [cout, oh, ow, num_groups]

        # output channel, output feature height, output feature width, num_groups
        cout, oh, ow, num_groups = num_eff_ops.shape
    
        iter_cout = math.ceil(cout / num_pe_row)
        iter_oh   = oh
        iter_ow   = math.ceil(ow / num_pe_col)
        for ti_oh in range(iter_oh):
            tile_oh = num_eff_ops[:, ti_oh, :, :] # [cout, ow, num_groups]
            for ti_cout in range(iter_cout):
                l_ti_cout = ti_cout * num_pe_row
                u_ti_cout = (ti_cout+1) * num_pe_row
                tile_cout = tile_oh[l_ti_cout:u_ti_cout, :, :] # [cout, ow, num_groups]
                for ti_ow in range(iter_ow):
                    l_ti_ow = ti_ow * num_pe_col
                    u_ti_ow = (ti_ow+1) * num_pe_col
                    tile_ow = tile_cout[:, l_ti_ow:u_ti_ow, :] # [cout, ow, num_groups]
                    for group_id in range(num_groups):
                        group_eff_ops = tile_ow[:, :, group_id]
                        cycle_group = torch.max(group_eff_ops)
                        cycle_group = max(cycle_group.item(), self.buffer_load_cycle)
                        total_cycle += int(cycle_group)
        return total_cycle
    
    def _calc_cycle_dwconv(self, layer_name):
        total_cycle = 0
        num_pe_col = self.pe_array_dim['w']
        num_eff_ops = self.num_eff_ops[layer_name] # dimension: [cout, oh, ow, num_groups]
        # output channel, output feature height, output feature width, num_groups
        cout, oh, ow, num_groups = num_eff_ops.shape

        iter_oh   = oh
        iter_ow   = math.ceil(ow / num_pe_col)
        for ti_oh in range(iter_oh):
            tile_oh = num_eff_ops[:, ti_oh, :, :] # [cout, ow, num_groups]
            for ti_cout in range(cout):
                tile_cout = tile_oh[ti_cout, :, :] # [ow, num_groups]
                for ti_ow in range(iter_ow):
                    l_ti_ow = ti_ow * num_pe_col
                    u_ti_ow = (ti_ow+1) * num_pe_col
                    tile_ow = tile_cout[l_ti_ow:u_ti_ow, :] # [cout, ow, num_groups]
                    for group_id in range(num_groups):
                        group_eff_ops = tile_ow[:, group_id]
                        cycle_group = torch.max(group_eff_ops)
                        cycle_group = max(cycle_group.item(), self.buffer_load_cycle / 2)
                        total_cycle += int(cycle_group)
        return total_cycle

    def _calc_cycle_fc(self, layer_name):
        total_cycle = 0
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']
        num_eff_ops = self.num_eff_ops[layer_name] # dimension: [sample size, cout, num_groups]

        # batch size, output channel, num_groups
        token_num, cout, num_groups = num_eff_ops.shape
        
        iter_cout  = math.ceil(cout / num_pe_row)
        iter_sample = math.ceil(token_num / num_pe_col)
        for ti_sample in range(iter_sample):
            l_ti_sample = ti_sample * num_pe_col
            u_ti_sample = (ti_sample+1) * num_pe_col
            tile_sample = num_eff_ops[l_ti_sample:u_ti_sample, :, :] # [sample, ow, num_groups]
            for ti_cout in range(iter_cout):
                l_ti_cout = ti_cout * num_pe_row
                u_ti_cout = (ti_cout+1) * num_pe_row
                tile_cout = tile_sample[:, l_ti_cout:u_ti_cout, :] # [sample, ow, num_groups]
                for group_id in range(num_groups):
                    group_eff_ops = tile_cout[:, :, group_id]
                    cycle_group = torch.max(group_eff_ops)
                    cycle_group = max(cycle_group.item(), self.buffer_load_cycle)
                    total_cycle += int(cycle_group)
        return total_cycle
    
    def _calc_dram_cycle(self):
        self._layer_cycle_dram = {}
        i_prec = self.input_precision
        w_prec = self.input_precision
        dram_bandwidth = self.dram.rw_bw * 2 # DDR
        size_sram_i = self.i_sram.size / 8
        self._read_layer_input = True

        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            num_dram_fetch_w, num_dram_fetch_i = self._layer_mem_refetch[name]
            cycle_layer_dram = 0
            if w_dim is not None:
                cycle_dram_load_w = self._w_mem_required[name] * w_prec / dram_bandwidth
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
        tile_cout   = math.ceil(cout / num_pe_row)
        tile_ow     = math.ceil(ow / num_pe_col)
        tile_oh     = oh

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
    
    def _calc_eff_ops(self):
        num_eff_ops = 0
        for name in self.layer_name_list:
            num_eff_ops += torch.sum(self.num_eff_ops[name]).item()
        self.total_eff_ops = num_eff_ops

    def calc_compute_energy(self, use_old=False):
        if not use_old:
            compute_energy = self.total_eff_ops * self.PE_ENERGY
        else:
            num_pe = self.total_pe_count
            if self.cycle_compute is None:
                self.cycle_compute, _ = self.calc_cycle()
            num_cycle_compute = self.cycle_compute
            compute_energy = self.PE_ENERGY * num_pe * num_cycle_compute
        return compute_energy
    
    def calc_local_buffer_rd_energy(self):
        buffer_rd_cost = self.local_buffer.r_cost / self.total_pe_count
        total_energy = buffer_rd_cost * self.total_eff_ops
        return total_energy
    
    def calc_local_buffer_wr_energy(self):
        buffer_wr_cost = self.local_buffer.w_cost_min
        w_buffer_reuse = self.LOCAL_OUTPUT_BUFFER_SIZE
        i_buffer_wr_cost = buffer_wr_cost / 2
        w_buffer_wr_cost = buffer_wr_cost / 2

        total_energy = 0
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    cin = i_dim[3]
                    cw  = w_dim[2]
                    if cin == cw: 
                        num_tile = self._calc_tile_conv2d(w_dim, o_dim)
                        total_energy += (i_buffer_wr_cost * num_tile) + \
                                        (w_buffer_wr_cost * num_tile / w_buffer_reuse)
                    else: # depthwise conv
                        num_tile = self._calc_tile_dwconv(w_dim, i_dim, o_dim)
                        total_energy += ( (i_buffer_wr_cost * num_tile) +
                                (w_buffer_wr_cost * num_tile / w_buffer_reuse) ) / self.pe_array_dim['h']
                else:
                    num_tile = self._calc_tile_fc(w_dim, o_dim)
                    total_energy += (i_buffer_wr_cost * num_tile) + \
                                    (w_buffer_wr_cost * num_tile / w_buffer_reuse)
        return total_energy
    
    def calc_sram_rd_energy(self):
        w_sram_rd_cost = self.w_sram.r_cost
        i_sram_rd_cost = self.i_sram.r_cost
        w_buffer_reuse = self.LOCAL_OUTPUT_BUFFER_SIZE
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']
        num_tile = self.calc_pe_array_tile()

        i_sram_rd_energy = i_sram_rd_cost * num_pe_col * num_tile
        w_sram_rd_energy = w_sram_rd_cost * num_pe_row * num_tile / w_buffer_reuse
        sram_rd_energy = i_sram_rd_energy + w_sram_rd_energy
        return sram_rd_energy
    
    def calc_sram_wr_energy(self):
        total_energy = 0
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            o_num_total = np.prod(o_dim)
            i_num_total = np.prod(i_dim)
            i_num_zero  = self.i_num_zero[name]
            o_num_zero  = self.o_num_zero[name]
            w_density   = 1
            i_density   = 1 - (i_num_zero / i_num_total)
            o_density   = 1 - (o_num_zero / o_num_total)
            if w_dim is not None:
                if len(w_dim) == 4:
                    total_energy += self._calc_sram_wr_energy_conv(name, w_dim, i_dim, o_dim, 
                                                                   w_density, i_density, o_density)
                else:
                    total_energy += self._calc_sram_wr_energy_fc(name, w_dim, i_dim, o_dim,
                                                                 w_density, i_density, o_density)
        return total_energy
    
    def _calc_sram_wr_energy_conv(self, layer_name, w_dim, i_dim, o_dim, w_density, i_density, o_density):
        w_prec = self.input_precision
        i_prec = self.input_precision
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
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * w_density * num_fetch_w

        num_i_sram_wr  = math.ceil(cin * i_prec / i_sram_min_wr_bw) * ih * iw 
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * i_density * num_fetch_i

        num_o_sram_wr  = math.ceil(cout * i_prec / i_sram_min_wr_bw) * oh * ow 
        energy_o_sram_wr = num_o_sram_wr * o_density * i_sram_wr_cost

        total_energy = energy_w_sram_wr + energy_i_sram_wr + energy_o_sram_wr
        return total_energy
    
    def _calc_sram_wr_energy_fc(self, layer_name, w_dim, i_dim, o_dim, w_density, i_density, o_density):
        w_prec = self.input_precision
        i_prec = self.input_precision
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
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * w_density * num_fetch_w

        num_i_sram_wr  = math.ceil(cin * i_prec / i_sram_min_wr_bw)  * token_num
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * i_density * num_fetch_i

        num_o_sram_wr  = math.ceil(cout * i_prec / i_sram_min_wr_bw)  * token_num
        if token_num == 1: # CNN last FC layer
            energy_o_sram_wr = 0
        else:
            energy_o_sram_wr = num_o_sram_wr * o_density * i_sram_wr_cost

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
    
    def _get_sparse_info(self, model_name, model, group_size, args):
        zero_ops_profiler = SpartenProfiler(model_name, model, group_size, args=args, device=self.DEVICE)
        zero_ops_profiler.fit()
        return (zero_ops_profiler.num_zero_input, 
                zero_ops_profiler.num_zero_output, 
                zero_ops_profiler.num_zero_weight, 
                zero_ops_profiler.num_eff_ops)
    
    def _check_layer_mem_size(self):
        self._w_mem_required = {}
        self._i_mem_required = {}
        self._o_mem_required = {}
        w_prec = self.input_precision
        i_prec = self.input_precision

        # scaling memory to store metadata, i.e. sparse map
        i_mem_scaling = (i_prec + 1) / i_prec 
        o_mem_scaling = i_mem_scaling 

        for layer_idx, name in enumerate(self.layer_name_list):
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            w_num_total = np.prod(w_dim)
            o_num_total = np.prod(o_dim)
            i_num_total = np.prod(i_dim)
            w_num_zero  = self.w_num_zero[name]
            i_num_zero  = self.i_num_zero[name]
            o_num_zero  = self.o_num_zero[name]
            w_density   = 1 - (w_num_zero / w_num_total)
            i_density   = 1 - (i_num_zero / i_num_total)
            o_density   = 1 - (o_num_zero / o_num_total)
            
            if w_dim is not None:
                if len(w_dim) == 4:
                    # kernel size, kernel input channel, output channel
                    k, _, cw, cout = w_dim
                    # batch size, input feature width, input feature height, input channel
                    batch_size, iw, ih, cin = i_dim
                    # batch size, output feature height, output feature width, output channel
                    _, oh ,ow, _ = o_dim

                    w_mem_dense = math.ceil(cw * w_prec / 8) * k**2 * cout
                    i_mem_dense = math.ceil(cin * i_prec / 8) * ih * iw 
                    o_mem_dense = math.ceil(cout * i_prec / 8) * oh * ow 
                    self._w_mem_required[name] = math.ceil(w_mem_dense * w_density / 8) * 8
                    self._i_mem_required[name] = math.ceil(i_mem_dense * i_density * i_mem_scaling / 8) * 8
                    self._o_mem_required[name] = math.ceil(o_mem_dense * o_density * o_mem_scaling / 8) * 8
                else:
                    # input channel, output channel
                    cin, cout = w_dim
                    # batch size, output channel
                    batch_size, token_num, _ = o_dim

                    w_mem_dense = math.ceil(cin * w_prec / 8) * cout
                    i_mem_dense = math.ceil(cin * i_prec / 8)  * token_num
                    if layer_idx == (len(self.layer_name_list) - 1):
                        o_mem_dense = 0
                    else:
                        o_mem_dense = math.ceil(cout * i_prec / 8)  * token_num
                    self._w_mem_required[name] = math.ceil(w_mem_dense * w_density / 8) * 8
                    self._i_mem_required[name] = math.ceil(i_mem_dense * i_density * i_mem_scaling / 8) * 8
                    self._o_mem_required[name] = math.ceil(o_mem_dense * o_density * o_mem_scaling / 8) * 8
    
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
        op_prec = self.input_precision
        group_size = self.pe_dotprod_size

        # local buffer 
        # every PE contains a local buffer with 128-byte weight, 128-byte input, 256-bit bitmask
        # every PE's local buffer can read out 1-byte weight, 1-byte input, 2-bit bitmask
        local_buffer_bank = self.pe_array_dim['h'] * self.pe_array_dim['w'] / op_prec * (op_prec+1) 
        local_buffer_config = {
                                'technology': 0.028,
                                'mem_type': 'sram', 
                                'size': 512*8 * local_buffer_bank, 
                                'bank_count': local_buffer_bank, 
                                'rw_bw': 2 * op_prec * local_buffer_bank,
                                'r_port': 1, 
                                'w_port': 1, 
                                'rw_port': 0,
                            }
        self.local_buffer = MemoryInstance('local_buffer', local_buffer_config, 
                                            r_cost=0, w_cost=0, latency=1, area=0, 
                                            min_r_granularity=None, 
                                            min_w_granularity=group_size * op_prec * local_buffer_bank, 
                                            get_cost_from_cacti=True, 
                                            double_buffering_support=False)
        #print(self.local_buffer.r_cost, self.local_buffer.w_cost_min)

        w_sram_bank = 16 // op_prec * (op_prec+1)  # one bank feeds 2 PE rows with value and metadata
        w_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 9 * 1024*8 * w_sram_bank, 
                            'bank_count': w_sram_bank, 
                            'rw_bw': group_size * (op_prec+1), 
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.w_sram = MemoryInstance('w_sram', w_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=None, min_w_granularity=64,  
                                     get_cost_from_cacti=True, double_buffering_support=False)
        
        i_sram_bank = 16 // op_prec * (op_prec+1) # one bank feeds 1 PE column with value and metadata
        i_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 8 * 1024*8 * i_sram_bank, 
                            'bank_count': i_sram_bank, 
                            'rw_bw': group_size * (op_prec+1),
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.i_sram = MemoryInstance('i_sram', i_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=None, min_w_granularity=64, 
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
     
        #w_sram_bw = self.w_sram.rw_bw
        i_sram_bw = self.i_sram.rw_bw
        #num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']
        #w_buffer_load_cycle = (group_size * (op_prec + 1) * num_pe_row) / w_sram_bw
        i_buffer_load_cycle = (group_size * (op_prec + 1) * num_pe_col) / i_sram_bw
        self.buffer_load_cycle = i_buffer_load_cycle
        #self.buffer_load_cycle = max(w_buffer_load_cycle, i_buffer_load_cycle)
