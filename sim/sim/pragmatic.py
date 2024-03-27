import math
import torch
import torch.nn as nn
import numpy as np

from typing import List
from hw.mem.mem_instance import MemoryInstance
from hw.alu.alu_unit import BitSerialPE
from hw.accelerator import Accelerator
from sim.model_quantized import MODEL

from sim.bin_int_convert import int_to_signMagnitude, int_to_twosComplement

# Pragmatic accelerator
class Pragmatic(Accelerator):

    PE_ENERGY = 0.4375 # energy per PE
    W_REG_ENERGY_PER_ROW = 1.01125 # energy (pJ) of the weight shift register file for a PE row
    I_REG_ENERGY_PER_COL = 0.53125 # energy (pJ) of the activation register file for a PE column
    PE_AREA = 1

    def __init__(self, 
                 input_precision_s: int, # bit-serial operand precision
                 input_precision_p: int, # bit-parallel operand precision
                 pe_dotprod_size: int, # length of the dot product inside one PE
                 pe_array_dim: List[int],
                 model_name: str,
                 model: nn.Module): # model comes from "BitSim/sim.model_profile/models/models.py
        assert len(pe_array_dim) == 2, \
            f'PE array must have 2 dimensions, but you gave {len(pe_array_dim)}'
        
        self.pe_array_dim = {'h': pe_array_dim[0], 'w': pe_array_dim[1]}
        self.pe_dotprod_size   = pe_dotprod_size
        
        pe = BitSerialPE(input_precision_s, input_precision_p, 
                         pe_dotprod_size, self.PE_ENERGY, self.PE_AREA)
        super().__init__(pe, self.pe_array_dim, model_name, model)
        self.model_q = MODEL[model_name].cpu() # quantized model
        self._init_mem()
    
    def calc_cycle(self):
        cycle = 0
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    cin = i_dim[3]
                    cw  = w_dim[2]
                    if cin == cw: 
                        cycle += self._calc_conv2d_cycle(name, w_dim, o_dim)
                    else: # depthwise conv
                        cycle += self._calc_dwconv_cycle(name, w_dim, i_dim, o_dim)
                else:
                    cycle += self._calc_fc_cycle(name, w_dim, o_dim)
        return cycle
    
    def _calc_conv2d_cycle(self, layer_name, w_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]
        wq_b = wq_b[1:, :, :, :, :]

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
                        if ( u_ti_k <= k**2 ):
                            tile_k = tile_cw[:, :, l_ti_k:u_ti_k]
                        else:
                            tile_k = tile_cw[:, :, l_ti_k:]
                        cycle_tile_k = torch.max(torch.sum(tile_k, dim=0))
                        cycle_kernel += int(cycle_tile_k.item())
            else:
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
                            cycle_tile_cw = torch.max(torch.sum(tile_cw, dim=0))
                            cycle_kernel += int(cycle_tile_cw.item())
        cycle_ow    = math.ceil(ow / num_pe_col)
        cycle_oh    = oh

        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch * batch_size
        return total_cycle
    
    def _calc_dwconv_cycle(self, layer_name, w_dim, i_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]
        wq_b = wq_b[1:, :, :, :, :]

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
                    cycle_tile_k = torch.max(torch.sum(tile_k, dim=0))
                    cycle_kernel += int(cycle_tile_k.item())
        cycle_ow    = math.ceil(ow / num_pe_col)
        cycle_oh    = oh

        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch * batch_size
        return total_cycle

    def _calc_fc_cycle(self, layer_name, w_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]
        wq_b = wq_b[1:, :, :]

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
                cycle_tile_cin = torch.max(torch.sum(tile_cin, dim=0))
                cycle_kernel += int(cycle_tile_cin.item())
        cycle_batch = math.ceil(batch_size / num_pe_col)

        total_cycle = cycle_kernel * cycle_batch
        return total_cycle
    
    def _get_quantized_weight(self, layer_name):
        for name, layer in self.model_q.named_modules():
            if ( layer_name == name ):
                w = layer.weight()
                wq = torch.int_repr(w)
                wqb_signMagnitude = int_to_signMagnitude(wq, w_bitwidth=8, device=self.DEVICE)
                if len(wqb_signMagnitude.shape) == 5:
                    wqb_signMagnitude = wqb_signMagnitude.permute([0, 1, 3, 4, 2])
                return wqb_signMagnitude
        raise Exception(f'ERROR! The layer {layer_name} cannot be found in the quantized model!')

    def num_mem_refetch(self, w_dim, i_dim):
        # If the on-chip buffer size is not big enough, 
        # we need to refetch input tiles or weight tiles from DRAM
        size_sram_w = self.w_sram.size / 8
        size_sram_i  = self.i_sram.size / 8
        w_mem_required = np.prod(w_dim) * self.pe.input_precision_p / 8
        i_mem_required = np.prod(i_dim) * self.pe.input_precision_p / 8
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
                return num_refetch_weight, 1 
            else:
                #print(f'Refetch input for {num_refetch_input} times ...')
                # refetch all input for every weight tile
                return 1, num_refetch_input
        else:
            # no need refetch
            return 1, 1
        
    def calc_compute_energy(self):
        w_prec = self.pe.input_precision_s
        num_pe = self.pe_array.total_unit_count
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']
        num_cycle = self.calc_cycle()

        pe_energy = self.PE_ENERGY * num_pe * num_cycle
        w_reg_energy = self.W_REG_ENERGY_PER_ROW * num_pe_row * num_cycle
        # The activation register is accessed every w_prec cycles
        i_reg_energy = self.I_REG_ENERGY_PER_COL * num_pe_col * num_cycle / w_prec
        compute_energy = pe_energy + w_reg_energy + i_reg_energy

        return compute_energy
    
    def calc_sram_rd_energy(self):
        w_prec = self.pe.input_precision_s
        w_sram_rd_cost = self.w_sram.r_cost
        i_sram_rd_cost = self.i_sram.r_cost
        num_cycle = self.calc_cycle()

        total_energy = math.ceil(num_cycle / w_prec) * (w_sram_rd_cost + i_sram_rd_cost)
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is None:
                total_energy += self._calc_residual_sram_energy(o_dim)
        return total_energy
    
    def calc_sram_wr_energy(self):
        energy = 0
        for layer_idx, name in enumerate(self.layer_name_list):
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                num_fetch_w, num_fetch_i = self.num_mem_refetch(w_dim, i_dim)
                if len(w_dim) == 4:
                    energy += self._calc_conv_sram_wr_energy(w_dim, i_dim,
                                                             num_fetch_w, num_fetch_i)
                else:
                    energy += self._calc_fc_sram_wr_energy(layer_idx, w_dim, o_dim, 
                                                           num_fetch_w, num_fetch_i)
        return energy
    
    def calc_dram_energy(self):
        energy = 0
        for layer_idx, name in enumerate(self.layer_name_list):
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                num_fetch_w, num_fetch_i = self.num_mem_refetch(w_dim, i_dim)
                if len(w_dim) == 4:
                    energy += self._calc_conv_dram_energy(w_dim, i_dim, o_dim, 
                                                          num_fetch_w, num_fetch_i)
                else:
                    energy += self._calc_fc_dram_energy(layer_idx, w_dim, o_dim, 
                                                        num_fetch_w, num_fetch_i)
            else:
                energy += self._calc_residual_dram_energy(o_dim)
        return energy
    
    def _calc_residual_sram_energy(self, o_dim):
        i_prec = self.pe.input_precision_p
        i_sram_rd_cost = self.i_sram.r_cost
        i_sram_wr_cost = self.i_sram.w_cost_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min

        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, cout = o_dim

        # energy_input: energy to write the residual map to SRAM, and then read out to DRAM
        num_i_sram_rw = math.ceil(ow * i_prec / i_sram_min_wr_bw) * oh * cout * batch_size 
        total_energy = num_i_sram_rw * (i_sram_rd_cost + i_sram_wr_cost)
        return total_energy
    
    def _calc_conv_sram_wr_energy(self, w_dim, i_dim, num_fetch_w: int=1, num_fetch_i: int=1):
        pe_group_size = self.pe_dotprod_size
        w_prec = self.pe.input_precision_p
        i_prec = self.pe.input_precision_p
        w_sram_wr_cost = self.w_sram.w_cost_min
        i_sram_wr_cost = self.i_sram.w_cost_min
        w_sram_min_wr_bw = self.w_sram.w_bw_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min

        # kernel size, kernel input channel, output channel
        batch_size, iw, ih, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        
        # write energy, read from DRAM and write to SRAM
        if ( k**2 > cw ):
            num_w_sram_wr = math.ceil((k**2) * w_prec / w_sram_min_wr_bw) * cw * cout
        else:
            num_w_sram_wr = math.ceil(cw * w_prec / w_sram_min_wr_bw) * (k**2) * cout
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * num_fetch_w

        num_i_sram_wr  = math.ceil(iw * i_prec / i_sram_min_wr_bw) * ih * cin * batch_size
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i

        total_energy = energy_w_sram_wr + energy_i_sram_wr
        return total_energy
    
    def _calc_fc_sram_wr_energy(self, layer_idx, w_dim, o_dim, num_fetch_w: int=1, num_fetch_i: int=1):
        w_prec = self.pe.input_precision_p
        i_prec = self.pe.input_precision_p
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
    
    def _calc_conv_dram_energy(self, w_dim, i_dim, o_dim, num_fetch_w: int=1, num_fetch_i: int=1):
        w_prec = self.pe.input_precision_s
        i_prec = self.pe.input_precision_p
        pe_group_size = self.pe_dotprod_size
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost
        wr_cost = self.dram.w_cost

        # kernel size, kernel input channel, output channel
        batch_size, iw, ih, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        _, ow, oh, _ = o_dim

        # energy_input:  energy to read input feature map
        # energy_output: energy to write output feature map
        # energy_weight: energy to read weight
        if ( k**2 > cw ):
            energy_weight = math.ceil((k**2) * w_prec / bus_width) * cw * rd_cost 
        else:
            energy_weight = math.ceil(cw * w_prec / bus_width) * (k**2) * rd_cost
        energy_input  = math.ceil(iw * i_prec / bus_width) * ih * cin * batch_size * rd_cost 
        energy_output = math.ceil(ow * i_prec / bus_width) * oh * cout * batch_size * wr_cost

        energy_weight *= num_fetch_w
        energy_input  *= num_fetch_i

        total_energy = energy_weight + energy_input + energy_output
        return total_energy
    
    def _calc_residual_dram_energy(self, o_dim):
        i_prec = self.pe.input_precision_p
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost

        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, cout = o_dim

        # energy_input: energy to read the residual map
        energy_input = math.ceil(ow * i_prec / bus_width) * oh * cout * batch_size * rd_cost

        total_energy = energy_input
        return total_energy
    
    def _calc_fc_dram_energy(self, layer_idx, w_dim, o_dim, num_fetch_w: int=1, num_fetch_i: int=1):
        w_prec = self.pe.input_precision_s
        i_prec = self.pe.input_precision_p
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost
        wr_cost = self.dram.w_cost

        # input channel, output channel
        cin, cout = w_dim
        # batch size, output channel
        batch_size, _ = o_dim

        # energy_input:  energy to read input feature map
        # energy_output: energy to write output feature map
        # energy_weight: energy to read weight
        energy_weight = math.ceil(cin * w_prec / bus_width) * cout * rd_cost
        energy_input  = math.ceil(cin * i_prec / bus_width) * batch_size * rd_cost
        if layer_idx != len(self.layer_name_list): # last layer, no need to write output feature
            energy_output = 0
        else:
            energy_output = math.ceil(cout * i_prec / bus_width) * cout * batch_size * wr_cost
        
        energy_weight *= num_fetch_w
        energy_input  *= num_fetch_i
        total_energy = energy_weight + energy_input + energy_output
        return total_energy
    
    def _init_mem(self):
        w_prec = self.pe.input_precision_p
        w_sram_bank = 16 # one bank feeds 2 PE rows
        w_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 9 * w_sram_bank * 1024*8, 
                            'bank_count': w_sram_bank, 
                            'rw_bw': (self.pe_array_dim['h'] * w_prec) * w_sram_bank, 
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.w_sram = MemoryInstance('w_sram', w_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=None, min_w_granularity=64, 
                                     get_cost_from_cacti=True, double_buffering_support=False)
        
        i_prec = self.pe.input_precision_p
        i_sram_bank = 16 # one bank feeds 2 PE columns
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
                                    r_cost=545, w_cost=560, latency=1, area=0, 
                                    min_r_granularity=64, min_w_granularity=64, 
                                    get_cost_from_cacti=False, double_buffering_support=False)
