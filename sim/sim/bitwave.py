import math, time
import torch
import torch.nn as nn

from typing import List, Dict
from sim.stripes import Stripes
from sim.util.model_quantized import MODEL

from sim.util.bin_int_convert import int_to_signMagnitude

# Bitwave accelerator
class Bitwave(Stripes):
    PR_SCALING = 1.3 # scaling factor to account for post placement and routing
    DISPATCHER_ENERGY_PER_COL = 0.072625 
    PE_ENERGY = 0.2575 * PR_SCALING # energy per 8-way DP PE, multiplied by 1.3 to account for post P&R
    #PE_ENERGY = 0.30625 * PR_SCALING
    W_SCHEDULER_ENERGY_PER_ROW = 0.06575 * PR_SCALING # energy (pJ) of the weight scheduler for a PE row
    PE_AREA = 1

    def __init__(self, 
                 input_precision_s: int, # bit-serial operand precision
                 input_precision_p: int, # bit-parallel operand precision
                 pe_dotprod_size: int, # length of the dot product inside one PE
                 pe_array_dim: List[int],
                 model_name: str,
                 model: nn.Module, # model comes from "BitSim/sim.model_profile/models/models.py
                 layer_prec: Dict, # the precision of every layer
                 en_bitflip: bool=False # whether enable bitflip
                 ): 
        self.en_bitflip = en_bitflip
        self.model_q = MODEL[model_name].cpu() # quantized model

        super().__init__(input_precision_s, input_precision_p, pe_dotprod_size, 
                         pe_array_dim, model_name, model)
        
        # to be modified later
        self.layer_prec = {}
        for name in self.layer_name_list:
            self.layer_prec[name] = input_precision_s
    
    def calc_cycle(self):
        self._calc_compute_cycle()
        self._calc_dram_cycle()
        total_cycle = 0
        total_cycle_compute = 0
        for name in self.layer_name_list:
            cycle_layer_compute = self._layer_cycle_compute[name]
            cycle_layer_dram    = self._layer_cycle_dram[name]
            print(cycle_layer_compute, cycle_layer_dram)
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
                        cycle_layer_compute = self._calc_cycle_conv2d(name, w_dim, o_dim)
                    else: # depthwise conv
                        cycle_layer_compute = self._calc_cycle_dwconv(name, w_dim, i_dim, o_dim)
                else:
                    cycle_layer_compute = self._calc_cycle_fc(name, w_dim, o_dim)
                self._layer_cycle_compute[name] = cycle_layer_compute
            '''
            else:
                self._layer_cycle_compute[name] = 0
            '''

    def _count_zero_bit_column(self, bit_tensor):
        # bit_tensor: [bit_significance, cout, group_size]
        # reduce along group_size
        num_eff_bit_per_group  = torch.sum(bit_tensor, dim=-1)
        is_not_zero_bit_column = num_eff_bit_per_group.gt(0.)
        # reduce along bit_significance
        num_bit_column_per_group = torch.sum(is_not_zero_bit_column, dim=0)
        num_cycle = torch.max(num_bit_column_per_group)
        return num_cycle.item()
    
    def _calc_cycle_conv2d(self, layer_name, w_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]
        wq_b = wq_b[1:, :, :, :, :]

        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature height, output feature width, output channel
        batch_size, oh, ow, _ = o_dim

        # cycle_kernel: number of cycles to process a kernel
        # cycle_ow:     number of cycles along output width
        # cycle_oh:     number of cycles along output height
        cycle_kernel = 0
        iter_cout  = math.ceil(cout / num_pe_row)
        for ti_cout in range(iter_cout):
            l_ti_cout = ti_cout * num_pe_row
            u_ti_cout = (ti_cout+1) * num_pe_row
            # get the tile along output channel: [bit_significance, tile_cout, k, k, cw]
            tile_cout = wq_b[:, l_ti_cout:u_ti_cout, :, :, :]
            if ( k**2 > cw ):
                iter_cw = cw
                for ti_cw in range(iter_cw):
                    # get the tile along kernel input channel: [bit_significance, tile_cout, k, k]
                    tile_cw = tile_cout[:, :, :, :, ti_cw]
                    tile_cw = tile_cw.flatten(start_dim=2, end_dim=3)
                    iter_k  = math.ceil(k**2 / pe_group_size)
                    for ti_k in range(iter_k):
                        l_ti_k = ti_k * pe_group_size
                        u_ti_k = (ti_k+1) * pe_group_size
                        tile_k = tile_cw[:, :, l_ti_k:u_ti_k]

                        if self.en_bitflip:
                            cycle_tile_k = self._count_zero_bit_column(tile_k)
                        else:
                            cycle_tile_k = self.layer_prec[layer_name] - 1

                        cycle_kernel += int(cycle_tile_k)
            else:
                iter_cw = math.ceil(cw / pe_group_size)
                for tk1 in range(k):
                    for tk2 in range(k):
                        # get the tile along kernel width and height: [bit_significance, tile_cout, cw]
                        tile_k = tile_cout[:, :, tk1, tk2, :]
                        for ti_cw in range(iter_cw):
                            l_ti_cw = ti_cw * pe_group_size
                            u_ti_cw = (ti_cw+1) * pe_group_size
                            tile_cw = tile_k[:, :, l_ti_cw:u_ti_cw]

                            if self.en_bitflip:
                                cycle_tile_cw = self._count_zero_bit_column(tile_cw)
                            else:
                                cycle_tile_cw = self.layer_prec[layer_name] - 1

                            cycle_kernel += int(cycle_tile_cw)
        cycle_ow = math.ceil(ow / num_pe_col)
        cycle_oh = oh

        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch * batch_size
        return total_cycle
    
    def _calc_cycle_dwconv(self, layer_name, w_dim, i_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]
        wq_b = wq_b[1:, :, :, :, :]

        pe_group_size = self.pe_dotprod_size
        num_pe_col = self.pe_array_dim['w']

        # kernel size, kernel input channel, output channel
        _, _, _, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature height, output feature width, output channel
        batch_size, oh, ow, _ = o_dim

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
                    tile_k = tile_cw[:, l_ti_k:u_ti_k]

                    if self.en_bitflip:
                        cycle_tile_k = self._count_zero_bit_column(tile_k)
                    else:
                        cycle_tile_k = self.layer_prec[layer_name] - 1

                    cycle_kernel += int(cycle_tile_k)
        cycle_ow = math.ceil(ow / num_pe_col)
        cycle_oh = oh

        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch * batch_size
        return total_cycle

    def _calc_cycle_fc(self, layer_name, w_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]
        wq_b = wq_b[1:, :, :]

        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # input channel, output channel
        cin, cout = w_dim
        # batch size, sample size, output channel
        batch_size, sample_size, _ = o_dim

        # cycle_kernel: number of cycles to process a kernel
        # cycle_ow:     number of cycles along output width
        # cycle_oh:     number of cycles along output height
        cycle_kernel = 0
        iter_cout  = math.ceil(cout / num_pe_row)
        for ti_cout in range(iter_cout):
            l_ti_cout = ti_cout * num_pe_row
            u_ti_cout = (ti_cout+1) * num_pe_row
            # get the tile along output channel: [bit_significance, tile_cout, cin]
            tile_cout = wq_b[:, l_ti_cout:u_ti_cout, :]
            
            iter_cin = math.ceil(cin / pe_group_size)
            for ti_cin in range(iter_cin):
                l_ti_cin = ti_cin * pe_group_size
                u_ti_cin = (ti_cin+1) * pe_group_size
                tile_cin = tile_cout[:, :, l_ti_cin:u_ti_cin]

                if not self.en_bitflip:
                    cycle_tile_cin = self._count_zero_bit_column(tile_cin)
                else:
                    cycle_tile_cin = self.layer_prec[layer_name] - 1

                cycle_kernel += int(cycle_tile_cin)
        cycle_batch = math.ceil(batch_size * sample_size / num_pe_col)

        total_cycle = cycle_kernel * cycle_batch
        return total_cycle
    
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
        total_tile = tile_per_batch * batch_size
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
        total_tile = tile_per_batch * batch_size
        return total_tile

    def _calc_tile_fc(self, w_dim, o_dim):
        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # input channel, output channel
        cin, cout = w_dim
        # batch size, sample size, output channel
        batch_size, sample_size, _ = o_dim

        # tile_in_channel:   number of tiles along input channel
        # tile_cout:  number of tiles along output channel
        tile_in_channel  = math.ceil(cin / pe_group_size)
        tile_cout        = math.ceil(cout / num_pe_row)
        tile_batch       = math.ceil(batch_size * sample_size / num_pe_col)

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
            '''
            else:
                cycle_layer_dram = self._calc_residual_dram_latency(o_dim)
            '''
            self._layer_cycle_dram[name] = int(cycle_layer_dram)
    
    def _calc_residual_dram_latency(self, o_dim):
        i_prec = self.pe.input_precision_p
        dram_bandwidth = self.dram.rw_bw * 2 # DDR # DDR
        # batch size, output feature height, output feature width, output channel
        batch_size, oh ,ow, cout = o_dim
        total_cycle = math.ceil(cout * i_prec / dram_bandwidth) * oh * ow * batch_size
        return total_cycle
    
    def calc_compute_energy(self):
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
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    total_energy += self._calc_sram_wr_energy_conv(name, w_dim, i_dim, o_dim)
                else:
                    total_energy += self._calc_sram_wr_energy_fc(name, w_dim, i_dim, o_dim)
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
            '''
            else:
                energy += self._calc_residual_dram_energy(o_dim)
            '''
        return energy
    
    def _calc_residual_sram_energy(self, o_dim):
        i_prec = self.pe.input_precision_p
        i_sram_rd_cost = self.i_sram.r_cost
        i_sram_wr_cost = self.i_sram.w_cost_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min

        # batch size, output feature height, output feature width, output channel
        batch_size, oh ,ow, cout = o_dim

        # energy_input: energy to write the residual map to SRAM, and then read out to DRAM
        num_i_sram_rw = math.ceil(cout * i_prec / i_sram_min_wr_bw) * oh * ow * batch_size 
        total_energy = num_i_sram_rw * (i_sram_rd_cost + i_sram_wr_cost)
        return total_energy
    
    def _calc_sram_wr_energy_conv(self, layer_name, w_dim, i_dim, o_dim):
        w_prec = self.pe.input_precision_p
        i_prec = self.pe.input_precision_p
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

        num_i_sram_wr  = math.ceil(cin * i_prec / i_sram_min_wr_bw) * ih * iw * batch_size
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i

        num_o_sram_wr  = math.ceil(cout * i_prec / i_sram_min_wr_bw) * oh * ow * batch_size
        energy_o_sram_wr = num_o_sram_wr * i_sram_wr_cost

        total_energy = energy_w_sram_wr + energy_i_sram_wr + energy_o_sram_wr
        return total_energy
    
    def _calc_sram_wr_energy_fc(self, layer_name, w_dim, i_dim, o_dim):
        w_prec = self.pe.input_precision_p
        i_prec = self.pe.input_precision_p
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
        _, sample_size, _ = o_dim

        # write energy, read from DRAM and write to SRAM
        num_w_sram_wr = math.ceil(cin * w_prec / w_sram_min_wr_bw) * cout
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * num_fetch_w

        num_i_sram_wr  = math.ceil(cin * i_prec / i_sram_min_wr_bw) * batch_size * sample_size
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i

        num_o_sram_wr  = math.ceil(cout * i_prec / i_sram_min_wr_bw) * batch_size * sample_size
        if sample_size == 1: # CNN last FC layer
            energy_o_sram_wr = 0
        else:
            energy_o_sram_wr = num_o_sram_wr * i_sram_wr_cost

        total_energy = energy_w_sram_wr + energy_i_sram_wr + energy_o_sram_wr
        return total_energy

    def _calc_dram_energy_conv(self, layer_name):
        i_prec = self.pe.input_precision_p
        w_prec = self.pe.input_precision_p
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
        i_prec = self.pe.input_precision_p
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost

        # batch size, output feature height, output feature width, output channel
        batch_size, oh ,ow, cout = o_dim

        # energy_input: energy to read the residual map
        energy_input = math.ceil(cout * i_prec / bus_width) * oh * ow * batch_size * rd_cost

        total_energy = energy_input
        return total_energy
    
    def _calc_dram_energy_fc(self, layer_name):
        w_prec = self.pe.input_precision_s
        i_prec = self.pe.input_precision_p
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
                    self._i_mem_required[name] = math.ceil(cin * i_prec / 8) * ih * iw * batch_size
                    self._o_mem_required[name] = math.ceil(cout * i_prec / 8) * oh * ow * batch_size
                else:
                    # input channel, output channel
                    cin, cout = w_dim
                    # batch size, sample size, output channel
                    batch_size, sample_size, _ = o_dim

                    self._w_mem_required[name] = math.ceil(cin * i_prec / 8) * cout * batch_size
                    self._i_mem_required[name] = math.ceil(cin * i_prec / 8) * batch_size * sample_size
                    if layer_idx == (len(self.layer_name_list) - 1):
                        self._o_mem_required[name] = 0
                    else:
                        self._o_mem_required[name] = math.ceil(cout * i_prec / 8) * batch_size * sample_size

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
    
    def _get_quantized_weight(self, layer_name):
        for name, layer in self.model_q.named_modules():
            if ( layer_name == name ):
                w = layer.weight()
                wq = torch.int_repr(w)
                wqb_twosComplement = int_to_signMagnitude(wq, w_bitwidth=8, device=self.DEVICE)
                if len(wqb_twosComplement.shape) == 5:
                    wqb_twosComplement = wqb_twosComplement.permute([0, 1, 3, 4, 2])
                return wqb_twosComplement
        raise Exception(f'ERROR! The layer {layer_name} cannot be found in the quantized model!')
