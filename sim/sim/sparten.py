import math
import torch
import torch.nn as nn
import numpy as np

from typing import List

from hw.accelerator import Accelerator
from hw.alu.alu_unit import PE
from hw.mem.mem_instance import MemoryInstance
from model_profile.meters.sparten_zero_ops import SpartenZeroOps

# Pragmatic accelerator
class Sparten(Accelerator):
    DISPATCHER_ENERGY_PER_COL = 0.072625
    PE_ENERGY = 1.11 # energy per PE, including MAC, prefix-sum, priority encoder
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
        self.pe_array_dim = {'h': pe_array_dim[0], 'w': pe_array_dim[1]}
        self.pe_dotprod_size = pe_dotprod_size

        pe = PE([input_precision, input_precision], 
                pe_dotprod_size, self.PE_ENERGY, self.PE_AREA)
        super().__init__(pe, self.pe_array_dim, model_name, model)

        self._init_mem()
        (self.i_num_zero, 
         self.o_num_zero, 
         self.w_num_zero, 
         self.num_zero_ops) = self._calc_zero_ops(model_name, model, args)
        self._check_layer_mem_size()
    
    def _calc_zero_ops(self, model_name, model, args):
        zero_ops_profiler = SpartenZeroOps(model_name, model, args=args, device=self.DEVICE)
        zero_ops_profiler.fit()
        return (zero_ops_profiler.num_zero_input, 
                zero_ops_profiler.num_zero_output, 
                zero_ops_profiler.num_zero_weight, 
                zero_ops_profiler.num_zero_ops)
    
    def calc_cycle(self):
        total_cycle = 0
        total_cycle_compute = 0
        for name in self.layer_name_list:
            cycle_layer_compute = self._layer_cycle_compute[name]
            cycle_layer_dram    = self._layer_cycle_dram[name]
            total_cycle_compute += cycle_layer_compute
            total_cycle += max(cycle_layer_compute, cycle_layer_dram)
        return total_cycle_compute, total_cycle
    
    def _calc_compute_cycle(self):
        self._layer_cycle_compute = {}
        for name in self.layer_name_list:
            cycle_layer_compute = 0
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    cin = i_dim[3]
                    cw  = w_dim[2]
                    if cin == cw: 
                        cycle_layer_compute = self._calc_conv2d_cycle(name, w_dim, o_dim)
                    else: # depthwise conv
                        cycle_layer_compute = self._calc_dwconv_cycle(name, w_dim, i_dim, o_dim)
                else:
                    cycle_layer_compute = self._calc_fc_cycle(name, w_dim, o_dim)
                self._layer_cycle_compute[name] = cycle_layer_compute
            '''
            else:
                self._layer_cycle_compute[name] = 0
            '''

    def _calc_conv2d_cycle(self, layer_name, w_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]

        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, _ = o_dim

        # cycle_kernel: number of cycles to process a kernel
        # cycle_ow:     number of cycles along output width
        # cycle_oh:     number of cycles along output height
        cycle_kernel = 0
        iter_cout  = math.ceil(cout / num_pe_row)
        for ti_cout in range(iter_cout):
            l_ti_cout = ti_cout * num_pe_row
            u_ti_cout = (ti_cout+1) * num_pe_row
            # get the tile along output channel: [bit_significance, tile_cout, k, k, cw]
            if ( u_ti_cout <= cout ):
                tile_cout = wq_b[:, l_ti_cout:u_ti_cout, :, :, :]
            else:
                tile_cout = wq_b[:, l_ti_cout:,          :, :, :]

            for tk1 in range(k):
                for tk2 in range(k):
                    # get the tile along kernel width and height: [bit_significance, tile_cout, cw]
                    tile_k = tile_cout[:, :, tk1, tk2, :]
                    iter_cw = math.ceil(cw / pe_group_size)
                    for ti_cw in range(iter_cw):
                        l_ti_cw = ti_cw * pe_group_size
                        u_ti_cw = (ti_cw+1) * pe_group_size
                        if ( u_ti_cw <= cw ):
                            tile_cw = tile_k[:, :, l_ti_cw:u_ti_cw]
                        else:
                            tile_cw = tile_k[:, :, l_ti_cw:]
                        cycle_tile_cw = torch.max(torch.sum(tile_cw, dim=-1))
                        cycle_kernel += int(cycle_tile_cw.item())
        cycle_ow    = math.ceil(ow / num_pe_col)
        cycle_oh    = oh

        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch * batch_size
        return total_cycle
    
    def _calc_dwconv_cycle(self, layer_name, w_dim, i_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]

        pe_group_size = self.pe_dotprod_size
        num_pe_col = self.pe_array_dim['w']

        # kernel size, kernel input channel, output channel
        _, _, _, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, _ = o_dim

        assert cin != cw, 'Not a depth-wise convolution!'

        # cycle_kernel: number of cycles to process a kernel
        # cycle_ow:     number of cycles along output width
        # cycle_oh:     number of cycles along output height
        cycle_kernel = 0
        iter_cout  = math.ceil(cout)
        for ti_cout in range(iter_cout):
            # get the tile along output channel: [bit_significance, tile_cout, k, k, cw]
            tile_cout = wq_b[:, ti_cout, :, :, :]
            iter_cw = cw
            for ti_cw in range(iter_cw):
                # get the tile along kernel input channel: [bit_significance, tile_cout, k, k]
                tile_cw = tile_cout[:, :, :, ti_cw]
                tile_cw = tile_cw.flatten(start_dim=1, end_dim=2)
                iter_k  = math.ceil(k**2 / pe_group_size)
                for ti_k in range(iter_k):
                    l_ti_k = ti_k * pe_group_size
                    u_ti_k = (ti_k+1) * pe_group_size
                    if ( u_ti_k <= k**2 ):
                        tile_k = tile_cw[:, l_ti_k:u_ti_k]
                    else:
                        tile_k = tile_cw[:, l_ti_k:]
                    cycle_tile_k = torch.max(torch.sum(tile_k, dim=-1))
                    cycle_kernel += int(cycle_tile_k.item())
        cycle_ow = math.ceil(ow / num_pe_col)
        cycle_oh = oh

        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch * batch_size
        return total_cycle

    def _calc_fc_cycle(self, layer_name, w_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]

        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # input channel, output channel
        cin, cout = w_dim
        # batch size, output channel
        batch_size, _ = o_dim

        # cycle_kernel: number of cycles to process a kernel
        # cycle_ow:     number of cycles along output width
        # cycle_oh:     number of cycles along output height
        cycle_kernel = 0
        iter_cout  = math.ceil(cout / num_pe_row)
        for ti_cout in range(iter_cout):
            l_ti_cout = ti_cout * num_pe_row
            u_ti_cout = (ti_cout+1) * num_pe_row
            # get the tile along output channel: [bit_significance, tile_cout, cin]
            if ( u_ti_cout <= cout ):
                tile_cout = wq_b[:, l_ti_cout:u_ti_cout, :]
            else:
                tile_cout = wq_b[:, l_ti_cout:,          :]
            
            iter_cin = math.ceil(cin / pe_group_size)
            for ti_cin in range(iter_cin):
                l_ti_cin = ti_cin * pe_group_size
                u_ti_cin = (ti_cin+1) * pe_group_size
                if ( u_ti_cin <= cin ):
                    tile_cin = tile_cout[:, :, l_ti_cin:u_ti_cin]
                else:
                    tile_cin = tile_cout[:, :, l_ti_cin:]
                cycle_tile_cin = torch.max(torch.sum(tile_cin, dim=-1))
                cycle_kernel += int(cycle_tile_cin.item())
        cycle_batch = math.ceil(batch_size / num_pe_col)
        total_cycle = cycle_kernel * cycle_batch
        return total_cycle
 
    def calc_compute_energy(self):
        num_pe = self.pe_array.total_unit_count
        num_cycle_compute = self.cycle_compute
        compute_energy = self.PE_ENERGY * num_pe * num_cycle_compute
        return compute_energy
    
    def _calc_dram_cycle(self):
        self._layer_cycle_dram = {}
        i_prec = self.input_precision
        w_prec_pad = self.input_precision
        dram_bandwidth = self.dram.rw_bw * 2
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
            '''
            else:
                cycle_layer_dram = self._calc_residual_dram_latency(o_dim)
            '''
            self._layer_cycle_dram[name] = int(cycle_layer_dram)
    
    def _calc_residual_dram_latency(self, o_dim):
        i_prec = self.input_precision
        dram_bandwidth = self.dram.rw_bw * 2 # DDR
        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, cout = o_dim
        total_cycle = math.ceil(cout * i_prec / dram_bandwidth) * oh * ow * batch_size
        return total_cycle
    
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
                    num_refetch_input = math.ceil(w_mem_required / size_sram_w)
                    num_refetch_weight = math.ceil(i_mem_required / size_sram_i)
                    total_fetch_weight = num_refetch_weight * w_mem_required
                    total_fetch_input = num_refetch_input * i_mem_required
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
        
    def calc_compute_energy(self):
        num_pe = self.pe_array.total_unit_count
        num_cycle_compute = self.cycle_compute
        compute_energy = self.PE_ENERGY * num_pe * num_cycle_compute
        return compute_energy
    
    def calc_sram_rd_energy(self):
        w_sram_rd_cost = self.w_sram.r_cost
        i_sram_rd_cost = self.i_sram.r_cost
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']
        num_cycle_compute = self.cycle_compute
        num_tile = self.calc_pe_array_tile()

        sram_rd_energy = num_tile * (w_sram_rd_cost + i_sram_rd_cost)
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is None:
                sram_rd_energy += self._calc_residual_sram_energy(o_dim)
        
        w_reg_energy = self.W_REG_ENERGY_PER_ROW * num_pe_row * num_cycle_compute
        # The activation register is accessed every tile
        i_reg_energy = self.I_REG_ENERGY_PER_COL * num_pe_col * num_tile
        total_energy = sram_rd_energy + w_reg_energy + i_reg_energy
        return total_energy
    
    def calc_sram_wr_energy(self):
        total_energy = 0
        for layer_idx, name in enumerate(self.layer_name_list):
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                num_fetch_w, num_fetch_i = self._layer_mem_refetch[name]
                if len(w_dim) == 4:
                    total_energy += self._calc_conv_sram_wr_energy(w_dim, i_dim, o_dim,
                                                             num_fetch_w, num_fetch_i)
                else:
                    total_energy += self._calc_fc_sram_wr_energy(layer_idx, w_dim, o_dim, 
                                                           num_fetch_w, num_fetch_i)
        return total_energy
    
    def _check_layer_mem_size(self):
        self._w_mem_required = {}
        self._i_mem_required = {}
        self._o_mem_required = {}
        w_prec = self.input_precision
        i_prec = self.input_precision

        # scaling memory to store metadata, i.e. sparse map
        w_mem_scaling = (w_prec + 1) / w_prec 
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
                    # batch size, output feature width, output feature height, output channel
                    _, ow, oh, _ = o_dim

                    w_mem_dense = math.ceil(cw * w_prec / 8) * k**2 * cout
                    i_mem_dense = math.ceil(cin * i_prec / 8) * ih * iw * batch_size
                    o_mem_dense = math.ceil(cout * i_prec / 8) * oh * ow * batch_size
                    self._w_mem_required[name] = w_mem_dense * w_density * w_mem_scaling
                    self._i_mem_required[name] = i_mem_dense * i_density * i_mem_scaling
                    self._o_mem_required[name] = o_mem_dense * o_density * o_mem_scaling
                else:
                    # input channel, output channel
                    cin, cout = w_dim
                    # batch size, output channel
                    batch_size, _ = o_dim

                    w_mem_dense = math.ceil(cin * i_prec / 8) * cout * batch_size
                    i_mem_dense = math.ceil(cin * i_prec / 8) * batch_size
                    if layer_idx == (len(self.layer_name_list) - 1):
                        o_mem_dense = 0
                    else:
                        o_mem_dense = math.ceil(cout * i_prec / 8) * batch_size
                    self._w_mem_required[name] = w_mem_dense * w_density * w_mem_scaling
                    self._i_mem_required[name] = i_mem_dense * i_density * i_mem_scaling
                    self._o_mem_required[name] = o_mem_dense * o_density * o_mem_scaling

    def calc_dram_energy(self):
        energy = 0
        self._read_layer_input = True
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    energy += self._calc_conv_dram_energy(name)
                else:
                    energy += self._calc_fc_dram_energy(name)
            '''
            else:
                energy += self._calc_residual_dram_energy(o_dim)
            '''
        return energy
    
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
                        total_tile += self._calc_conv2d_tile(w_dim, o_dim)
                    else: # depthwise conv
                        total_tile += self._calc_dwconv_tile(w_dim, i_dim, o_dim)
                else:
                    total_tile += self._calc_fc_tile(w_dim, o_dim)
        return total_tile
    
    def _calc_conv2d_tile(self, w_dim, o_dim):
        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, _ = o_dim

        # tile_kernel: number of tiles to process a kernel
        # tile_cout:   number of tiles along output channel
        # tile_ow:     number of tiles along output width
        # tile_oh:     number of tiles along output height
        '''
        if ( k**2 > cw ):
            tile_kernel   = math.ceil((k**2) / pe_group_size) * cw
        else:
            tile_kernel   = math.ceil(cw / pe_group_size) * (k**2)
        '''
        tile_kernel   = math.ceil(cw / pe_group_size) * (k**2)
        tile_cout  = math.ceil(cout / num_pe_row)
        tile_ow    = math.ceil(ow / num_pe_col)
        tile_oh    = oh

        tile_per_batch = (tile_kernel * tile_cout * tile_ow * tile_oh)
        total_tile = tile_per_batch * batch_size
        return total_tile
    
    def _calc_dwconv_tile(self, w_dim, i_dim, o_dim):
        pe_group_size = self.pe_dotprod_size
        num_pe_col = self.pe_array_dim['w']

        # kernel size, kernel input channel, output channel
        _, _, _, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, _ = o_dim

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
        total_tile = tile_per_batch * batch_size
        return total_tile

    def _calc_fc_tile(self, w_dim, o_dim):
        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # input channel, output channel
        cin, cout = w_dim
        # batch size, output channel
        batch_size, _ = o_dim

        # tile_in_channel:   number of tiles along input channel
        # tile_cout:  number of tiles along output channel
        tile_in_channel  = math.ceil(cin / pe_group_size)
        tile_cout        = math.ceil(cout / num_pe_row)
        tile_batch       = math.ceil(batch_size / num_pe_col)

        total_tile = (tile_in_channel * tile_cout * tile_batch)
        return total_tile
    
    def _calc_residual_sram_energy(self, o_dim):
        i_prec = self.input_precision
        i_sram_rd_cost = self.i_sram.r_cost
        i_sram_wr_cost = self.i_sram.w_cost_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min

        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, cout = o_dim

        # energy_input: energy to write the residual map to SRAM, and then read out to DRAM
        num_i_sram_rw = math.ceil(cout * i_prec / i_sram_min_wr_bw) * oh * ow * batch_size 
        total_energy = num_i_sram_rw * (i_sram_rd_cost + i_sram_wr_cost)
        return total_energy
    
    def _calc_conv_sram_wr_energy(self, w_dim, i_dim, o_dim, num_fetch_w: int=1, num_fetch_i: int=1):
        w_prec = self.input_precision
        i_prec = self.input_precision
        w_sram_wr_cost = self.w_sram.w_cost_min
        i_sram_wr_cost = self.i_sram.w_cost_min
        w_sram_min_wr_bw = self.w_sram.w_bw_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min

        # kernel size, kernel input channel, output channel
        batch_size, iw, ih, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        _, ow, oh, _ = o_dim
        
        # write energy, read from DRAM and write to SRAM
        num_w_sram_wr = math.ceil(cw * w_prec / w_sram_min_wr_bw) * (k**2) * cout
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * num_fetch_w

        num_i_sram_wr  = math.ceil(cin * i_prec / i_sram_min_wr_bw) * ih * iw * batch_size
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i

        num_o_sram_wr  = math.ceil(cout * i_prec / i_sram_min_wr_bw) * oh * ow * batch_size
        energy_o_sram_wr = num_o_sram_wr * i_sram_wr_cost

        total_energy = energy_w_sram_wr + energy_i_sram_wr + energy_o_sram_wr
        return total_energy
    
    def _calc_fc_sram_wr_energy(self, layer_idx, w_dim, o_dim, num_fetch_w: int=1, num_fetch_i: int=1):
        w_prec = self.input_precision
        i_prec = self.input_precision
        w_sram_wr_cost = self.w_sram.w_cost_min
        i_sram_wr_cost = self.i_sram.w_cost_min
        w_sram_min_wr_bw = self.w_sram.w_bw_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min

        # input channel, output channel
        cin, cout = w_dim
        # batch size, output channel
        batch_size, _ = o_dim

        # write energy, read from DRAM and write to SRAM
        num_w_sram_wr = math.ceil(cin * w_prec / w_sram_min_wr_bw) * cout
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * num_fetch_w

        num_i_sram_wr  = math.ceil(cin * i_prec / i_sram_min_wr_bw) * batch_size
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i

        total_energy = energy_w_sram_wr + energy_i_sram_wr
        return total_energy

    def _calc_conv_dram_energy(self, layer_name):
        i_prec = self.input_precision
        w_prec = self.input_precision
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost
        wr_cost = self.dram.w_cost
        size_sram_i = self.i_sram.size / 8
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]
        
        # energy_input:  energy to read input feature map
        # energy_output: energy to write output feature map
        # energy_weight: energy to read weight
        w_mem_required = self._w_mem_required[layer_name]
        energy_weight = w_mem_required * w_prec / bus_width * rd_cost

        i_mem_required = self._i_mem_required[layer_name]
        if self._read_layer_input:
            energy_input = i_mem_required * i_prec / bus_width * rd_cost 
        else:
            energy_input = 0

        o_mem_required = self._o_mem_required[layer_name]
        if ( i_mem_required + o_mem_required ) < size_sram_i:
            energy_output = 0
            self._read_layer_input = False
        else:
            energy_output = o_mem_required * i_prec / bus_width * wr_cost
            self._read_layer_input = True
        
        energy_weight *= num_fetch_w
        energy_input  *= num_fetch_i
        total_energy = energy_weight + energy_input + energy_output
        return total_energy
    
    def _calc_residual_dram_energy(self, o_dim):
        i_prec = self.input_precision
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost

        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, cout = o_dim

        # energy_input: energy to read the residual map
        energy_input = math.ceil(cout * i_prec / bus_width) * oh * ow * batch_size * rd_cost

        total_energy = energy_input
        return total_energy
    
    def _calc_fc_dram_energy(self, layer_name):
        w_prec = self.input_precision
        i_prec = self.input_precision
        size_sram_i = self.i_sram.size / 8
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost
        wr_cost = self.dram.w_cost
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        # energy_weight: energy to read weight from DRAM
        w_mem_required = self._w_mem_required[layer_name]
        energy_weight = w_mem_required * w_prec / bus_width * rd_cost
        
        # energy_input:  energy to read input feature map from DRAM
        i_mem_required = self._i_mem_required[layer_name]
        if self._read_layer_input:
            energy_input  = i_mem_required * i_prec / bus_width * rd_cost
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
                energy_output = o_mem_required * i_prec / bus_width * wr_cost
                self._read_layer_input = True

        energy_weight *= num_fetch_w
        energy_input  *= num_fetch_i
        total_energy = energy_weight + energy_input + energy_output
        return total_energy
    
    def _init_mem(self):
        local_buffer_bank = self.pe_array_dim['h'] * self.pe_array_dim['w']
        local_buffer_config = {
                                'technology': 0.028,
                                'mem_type': 'sram', 
                                'size': 512*8 * local_buffer_bank, 
                                'bank_count': local_buffer_bank, 
                                'rw_bw': 64 * local_buffer_bank,
                                'r_port': 1, 
                                'w_port': 1, 
                                'rw_port': 0,
                            }
        self.local_buffer = MemoryInstance('local_buffer', local_buffer_config, 
                                            r_cost=0, w_cost=0, latency=1, area=0, 
                                            min_r_granularity=16, 
                                            min_w_granularity=64, 
                                            get_cost_from_cacti=True, 
                                            double_buffering_support=False)
        #print(self.local_buffer.r_cost, self.local_buffer.w_cost_min)

        w_prec = self.input_precision
        w_sram_bank = 16 // w_prec * (w_prec+1)  # one bank feeds 2 PE rows with value and metadata
        w_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 9 * 1024*8 * w_sram_bank, 
                            'bank_count': w_sram_bank, 
                            'rw_bw': (self.pe_array_dim['h'] * w_prec) * w_sram_bank, 
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.w_sram = MemoryInstance('w_sram', w_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=None, 
                                     min_w_granularity=self.pe_array_dim['h'] * (w_prec+1),  
                                     get_cost_from_cacti=True, double_buffering_support=False)
        
        i_prec = self.input_precision
        i_sram_bank = 16 // i_prec * (i_prec+1) # one bank feeds 1 PE column with value and metadata
        i_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 8 * 1024*8 * i_sram_bank, 
                            'bank_count': i_sram_bank, 
                            'rw_bw': (self.pe_array_dim['w'] * i_prec) * i_sram_bank,
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.i_sram = MemoryInstance('i_sram', i_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=72, 
                                     min_w_granularity=self.pe_array_dim['w'] * (i_prec+1), 
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
     