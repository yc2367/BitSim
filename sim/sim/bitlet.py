import math
import torch
import torch.nn as nn
import numpy as np

from typing import List

from hw.mem.mem_instance import MemoryInstance
from sim.stripes import Stripes
from sim.util.bin_int_convert import int_to_twosComplement

# Bitlet accelerator
class Bitlet(Stripes):
    PR_SCALING = 1.5 # scaling factor to account for post placement and routing
    DISPATCHER_ENERGY_PER_COL = 0.072625
    #PE_ENERGY = 0.32 * PR_SCALING # energy per PE
    PE_ENERGY = 0.2925 * PR_SCALING # energy per PE
    W_REG_ENERGY_PER_ROW = 1.1325 * PR_SCALING # energy (pJ) of the weight shift register file for a PE row
    #W_REG_ENERGY_PER_ROW = 0.61875 * PR_SCALING
    I_REG_ENERGY_PER_COL = (1.054 + DISPATCHER_ENERGY_PER_COL) * PR_SCALING # energy (pJ) of the activation register file for a PE column
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
        self.min_intra_pe_op = 0 # The min number of bit ops in a weight group
        self.max_intra_pe_op = 0 # The max number of bit ops in a weight group that contains the min_intra_pe_op
        self.num_eff_op = 0
        self.num_total_op = 0
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

    def _calc_cycle_conv2d(self, layer_name, w_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]

        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        batch_size, oh ,ow, _ = o_dim

        # cycle_kernel: number of cycles to process a kernel
        # cycle_ow:     number of cycles along output width
        # cycle_oh:     number of cycles along output height
        cycle_kernel = 0
        num_eff_op_kernel = 0
        num_total_op_kernel = 0
        min_intra_pe_op_kernel = 0
        max_intra_pe_op_kernel = 0

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
                        
                        num_eff_bit_per_lane = torch.sum(tile_k, dim=-1)
                        cycle_tile_k = torch.max(num_eff_bit_per_lane).item()
                        cycle_kernel += int(cycle_tile_k)

                        min_intra_pe_op_kernel += torch.min(num_eff_bit_per_lane, dim=0).values.sum().item() * self.pe.input_precision_p
                        max_intra_pe_op_kernel += torch.max(num_eff_bit_per_lane, dim=0).values.sum().item() * self.pe.input_precision_p
                        num_eff_op_kernel += torch.sum(tile_k).item()
                        num_total_op_kernel += (cycle_tile_k * self.pe.input_precision_p * num_pe_row)
            else:
                for tk1 in range(k):
                    for tk2 in range(k):
                        # get the tile along kernel width and height: [bit_significance, tile_cout, cw]
                        tile_k = tile_cout[:, :, tk1, tk2, :]
                        iter_cw = math.ceil(cw / pe_group_size)
                        for ti_cw in range(iter_cw):
                            l_ti_cw = ti_cw * pe_group_size
                            u_ti_cw = (ti_cw+1) * pe_group_size
                            tile_cw = tile_k[:, :, l_ti_cw:u_ti_cw]

                            num_eff_bit_per_lane = torch.sum(tile_cw, dim=-1)
                            cycle_tile_cw = torch.max(num_eff_bit_per_lane).item()
                            cycle_kernel += int(cycle_tile_cw)

                            min_intra_pe_op_kernel += torch.min(num_eff_bit_per_lane, dim=0).values.sum().item() * self.pe.input_precision_p
                            max_intra_pe_op_kernel += torch.max(num_eff_bit_per_lane, dim=0).values.sum().item() * self.pe.input_precision_p
                            num_eff_op_kernel += torch.sum(tile_cw).item()
                            num_total_op_kernel += (cycle_tile_cw * self.pe.input_precision_p * num_pe_row)
        cycle_ow = math.ceil(ow / num_pe_col)
        cycle_oh = oh
        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch 

        min_intra_pe_op = min_intra_pe_op_kernel * cycle_ow * cycle_oh 
        max_intra_pe_op = max_intra_pe_op_kernel * cycle_ow * cycle_oh 
        num_eff_op = num_eff_op_kernel * cycle_ow * cycle_oh 
        num_total_op = num_total_op_kernel * cycle_ow * cycle_oh 
        self.num_eff_op += num_eff_op
        self.num_total_op += num_total_op
        self.min_intra_pe_op += min_intra_pe_op
        self.max_intra_pe_op += max_intra_pe_op
        return total_cycle
    
    def _calc_cycle_dwconv(self, layer_name, w_dim, i_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]

        pe_group_size = self.pe_dotprod_size
        num_pe_col = self.pe_array_dim['w']

        # kernel size, kernel input channel, output channel
        _, _, _, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        batch_size, oh ,ow, _ = o_dim

        assert cin != cw, 'Not a depth-wise convolution!'

        # cycle_kernel: number of cycles to process a kernel
        # cycle_ow:     number of cycles along output width
        # cycle_oh:     number of cycles along output height
        cycle_kernel = 0
        num_eff_op_kernel = 0
        num_total_op_kernel = 0

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
                    cycle_tile_k = torch.max(torch.sum(tile_k, dim=-1))
                    cycle_kernel += int(cycle_tile_k.item())

                    num_eff_op_kernel += torch.sum(tile_k).item()
                    num_total_op_kernel += (cycle_tile_k * self.pe.input_precision_p)
        cycle_ow = math.ceil(ow / num_pe_col)
        cycle_oh = oh
        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch 

        num_eff_op = num_eff_op_kernel * cycle_ow * cycle_oh 
        num_total_op = num_total_op_kernel * cycle_ow * cycle_oh 
        self.num_eff_op += num_eff_op
        self.num_total_op += num_total_op
        return total_cycle

    def _calc_cycle_fc(self, layer_name, w_dim, o_dim):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]

        pe_group_size = self.pe_dotprod_size
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']

        # input channel, output channel
        cin, cout = w_dim
        # batch size, sample size, output channel
        batch_size, token_num, _ = o_dim

        # cycle_kernel: number of cycles to process a kernel
        # num_eff_op:   number of effective operations performed
        # num_total_op: number of total operations performed
        cycle_kernel = 0
        num_eff_op_kernel = 0
        num_total_op_kernel = 0
        min_intra_pe_op_kernel = 0
        max_intra_pe_op_kernel = 0

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

                num_eff_bit_per_lane = torch.sum(tile_cin, dim=-1)
                cycle_tile_cin = torch.max(num_eff_bit_per_lane).item()
                cycle_kernel += int(cycle_tile_cin)

                min_intra_pe_op_kernel += torch.min(num_eff_bit_per_lane, dim=0).values.sum().item() * self.pe.input_precision_p
                max_intra_pe_op_kernel += torch.max(num_eff_bit_per_lane, dim=0).values.sum().item() * self.pe.input_precision_p
                num_eff_op_kernel += torch.sum(tile_cin).item()
                num_total_op_kernel += (cycle_tile_cin * self.pe.input_precision_p * num_pe_row)

        cycle_batch = math.ceil(token_num / num_pe_col)
        total_cycle = cycle_kernel * cycle_batch

        min_intra_pe_op = min_intra_pe_op_kernel * cycle_batch
        max_intra_pe_op = max_intra_pe_op_kernel * cycle_batch
        num_eff_op = num_eff_op_kernel * cycle_batch
        num_total_op = num_total_op_kernel * cycle_batch
        self.num_eff_op += num_eff_op
        self.num_total_op += num_total_op
        self.min_intra_pe_op += min_intra_pe_op
        self.max_intra_pe_op += max_intra_pe_op
        return total_cycle
    
    def _get_quantized_weight(self, layer_name):
        for name, wq in self.model_q.items():
            if ( layer_name == name ):
                wqb_twosComplement = int_to_twosComplement(wq, w_bitwidth=8, device=self.DEVICE)
                if len(wqb_twosComplement.shape) == 5:
                    wqb_twosComplement = wqb_twosComplement.permute([0, 1, 3, 4, 2])
                return wqb_twosComplement
        raise Exception(f'ERROR! The layer {layer_name} cannot be found in the quantized model!')
        
    def calc_compute_energy(self):
        num_pe = self.total_pe_count
        if self.cycle_compute is None:
            self.cycle_compute, _ = self.calc_cycle()
        num_cycle_compute = self.cycle_compute
        compute_energy = self.PE_ENERGY * num_pe * num_cycle_compute
        return compute_energy
    
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
                                     min_r_granularity=None, min_w_granularity=64,  
                                     get_cost_from_cacti=True, double_buffering_support=False)
        
        i_prec = self.pe.input_precision_p
        i_sram_bank = self.pe_array_dim['w'] # one bank feeds 1 PE column
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
     