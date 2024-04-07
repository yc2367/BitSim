import math, time
import torch
import torch.nn as nn

from typing import List, Dict
from sim.stripes import Stripes
from hw.mem.mem_instance import MemoryInstance
from hw.alu.alu_unit import BitSerialPE

from sim.util.model_quantized import MODEL
from sim.util.bin_int_convert import int_to_twosComplement

# Bitwave accelerator
class BitVert(Stripes):
    PR_SCALING = 1.5 # scaling factor to account for post placement and routing
    DISPATCHER_ENERGY_PER_COL = 0.072625 
    PE_ENERGY = 0.41 * PR_SCALING # energy per 32-way DP PE, multiplied by 1.5 to account for post P&R
    #PE_ENERGY = 0.28 * PR_SCALING # energy per 16-way DP PE, multiplied by 1.5 to account for post P&R
    #PE_ENERGY = 0.30625 * PR_SCALING
    W_REG_ENERGY_PER_ROW = 1.205 * PR_SCALING # energy (pJ) of the weight scheduler for a PE row
    I_REG_ENERGY_PER_COL = (1.06 + DISPATCHER_ENERGY_PER_COL) * PR_SCALING # energy (pJ) of the activation register file for a PE column

    PE_AREA = 1

    def __init__(self, 
                 input_precision_s: int, # bit-serial operand precision
                 input_precision_p: int, # bit-parallel operand precision
                 pe_dotprod_size: int, # length of the dot product inside one PE
                 pe_array_dim: List[int],
                 model_name: str,
                 model: nn.Module, # model comes from "BitSim/sim.model_profile/models/models.py
                 layer_prec:     Dict={},    # the precision of every layer
                 en_b2s:         bool=False,
                 en_lsb_pruning: bool=False, # whether enable LSB column pruning using bi-directional bit sparsity
                 en_ol_channel:  bool=False  # whether enable outlier channel groupping
                 ): 
        super().__init__(input_precision_s, input_precision_p, pe_dotprod_size, 
                         pe_array_dim, model_name, model, init_mem=False)
        
        self.model_q = MODEL[model_name].cpu() # quantized model
        
        if en_lsb_pruning or en_ol_channel:
            self.en_b2s = True
        else:
            self.en_b2s = en_b2s
        if en_ol_channel:
            self.en_lsb_pruning = False
        else: 
            self.en_lsb_pruning = en_lsb_pruning
        self.en_ol_channel = en_ol_channel

        # to be modified later
        self.w_prec_config = {}
        for name in self.layer_name_list:
            # format for a layer configuration: [(prec_high, % of channel), (prec_low, % of channel)]
            if (not self.en_lsb_pruning) and (not self.en_ol_channel):
                # assign input_precision_p to every layer
                self.w_prec_config[name] = [(input_precision_p, 1.0), (input_precision_p, 0.)]
            elif self.en_lsb_pruning:
                # only assign a single (lower) precision to every layer
                self.w_prec_config[name] = [(input_precision_p-2, 1.0), (input_precision_p, 0.)]
            elif self.en_ol_channel:
                # assign high precision to few outlier channels, assign low precision to most other channels
                self.w_prec_config[name] = [(input_precision_p, 0.25), (input_precision_p/2, 0.75)]
        # effective weight precision
        self.w_prec_eff = {}
        for name in self.layer_name_list:
            (prec_high, prec_high_ratio) = self.w_prec_config[name][0]
            (prec_low,  prec_low_ratio)  = self.w_prec_config[name][1]
            self.w_prec_eff[name] = prec_high*prec_high_ratio + prec_low*prec_low_ratio

        self._init_mem()
        self._check_layer_mem_size()
        self._calc_num_mem_refetch()

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
        for name in self.layer_name_list:
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

    def _count_column_msb_clip(self, bit_tensor):
        # bit_tensor: [bit_significance, cout, group_size]
        w_prec = self.pe.input_precision_p
        num_group = bit_tensor.shape[1]
        
        num_bit_column = torch.full([num_group], w_prec, device=self.DEVICE)
        eq_msb_column = torch.ones([num_group], dtype=torch.bool, device=self.DEVICE)
        for i in range(1, w_prec-4):
            eq_column = torch.all(torch.eq(bit_tensor[0], bit_tensor[i]), dim=-1)
            eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
            num_bit_column[eq_msb_column] = num_bit_column[eq_msb_column] - 1
        num_cycle = torch.max(num_bit_column)
        return num_cycle.item()
    
    def _count_no_b2s_cycle(self, bit_tensor):
        # bit_tensor: [bit_significance, cout, group_size]
        num_multiplier = self.pe_dotprod_size / 2

        num_eff_bit_per_column = torch.sum(bit_tensor, dim=-1)
        num_cycle_per_column = torch.ceil(num_eff_bit_per_column / num_multiplier)
        num_cycle_per_cout = torch.sum(num_cycle_per_column, dim=0)
        num_cycle = torch.max(num_cycle_per_cout).item()
        if num_cycle == 0:
            num_cycle = 1
        return num_cycle
    
    def _calc_cycle_conv2d(self, layer_name, w_dim, o_dim):
        (prec_high, prec_high_ratio) = self.w_prec_config[layer_name][0]
        (prec_low,  prec_low_ratio)  = self.w_prec_config[layer_name][1]
        num_tile = self._calc_tile_conv2d(w_dim, o_dim)

        if self.en_ol_channel:
            cycle_prec_high = num_tile * prec_high_ratio * prec_high
            cycle_prec_low  = num_tile * prec_low_ratio * prec_low
            return cycle_prec_high + cycle_prec_low
        elif self.en_lsb_pruning:
            return num_tile * prec_high
        
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name)
        pe_group_size = self.pe_dotprod_size
        num_pe_row    = self.pe_array_dim['h']
        num_pe_col    = self.pe_array_dim['w']

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
                        if not self.en_b2s:
                            cycle_tile_k = self._count_no_b2s_cycle(tile_k)
                        else:
                            cycle_tile_k = prec_high
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
                            if not self.en_b2s:
                                cycle_tile_cw = self._count_no_b2s_cycle(tile_cw)
                            else:
                                cycle_tile_cw = prec_high
                            cycle_kernel += int(cycle_tile_cw)
        cycle_ow = math.ceil(ow / num_pe_col)
        cycle_oh = oh
        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch * batch_size
        return total_cycle
    
    def _calc_cycle_dwconv(self, layer_name, w_dim, i_dim, o_dim):
        (prec_high, prec_high_ratio) = self.w_prec_config[layer_name][0]
        (prec_low,  prec_low_ratio)  = self.w_prec_config[layer_name][1]
        num_tile = self._calc_tile_dwconv(w_dim, i_dim, o_dim)

        if self.en_ol_channel:
            cycle_prec_high = num_tile * prec_high_ratio * prec_high
            cycle_prec_low  = num_tile * prec_low_ratio * prec_low
            return cycle_prec_high + cycle_prec_low
        elif self.en_lsb_pruning:
            return num_tile * prec_high
        
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]
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
                    if not self.en_b2s:
                        cycle_tile_k = self._count_no_b2s_cycle(tile_k)
                    else:
                        cycle_tile_k = prec_high
                    cycle_kernel += int(cycle_tile_k)
        cycle_ow = math.ceil(ow / num_pe_col)
        cycle_oh = oh
        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch * batch_size
        return total_cycle

    def _calc_cycle_fc(self, layer_name, w_dim, o_dim):
        (prec_high, prec_high_ratio) = self.w_prec_config[layer_name][0]
        (prec_low,  prec_low_ratio)  = self.w_prec_config[layer_name][1]
        num_tile = self._calc_tile_fc(w_dim, o_dim)

        if self.en_ol_channel:
            cycle_prec_high = num_tile * prec_high_ratio * prec_high
            cycle_prec_low  = num_tile * prec_low_ratio * prec_low
            return cycle_prec_high + cycle_prec_low
        elif self.en_lsb_pruning:
            return num_tile * prec_high
        
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
        # cycle_ow:     number of cycles along output width
        # cycle_oh:     number of cycles along output height
        cycle_kernel = 0
        iter_cout = math.ceil(cout / num_pe_row)
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
                if not self.en_b2s:
                    cycle_tile_cin = self._count_no_b2s_cycle(tile_cin)
                else:
                    cycle_tile_cin = prec_high
                cycle_kernel += int(cycle_tile_cin)
        cycle_batch = math.ceil(batch_size * token_num / num_pe_col)
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
    
    def _calc_dram_cycle(self):
        self._layer_cycle_dram = {}
        i_prec = self.pe.input_precision_p
        dram_bandwidth = self.dram.rw_bw * 2 # DDR
        size_sram_i = self.i_sram.size / 8
        self._read_layer_input = True

        for name in self.layer_name_list:
            w_prec = self.w_prec_eff[name]
            w_dim  = self.weight_dim[name]
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
    
    def calc_compute_energy(self):
        num_pe = self.total_pe_count
        if self.cycle_compute is None:
            self.cycle_compute, _ = self.calc_cycle()
        num_cycle_compute = self.cycle_compute
        compute_energy = self.PE_ENERGY * num_pe * num_cycle_compute
        return compute_energy
    
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

    def _check_layer_mem_size(self):
        self._w_mem_required = {}
        self._i_mem_required = {}
        self._o_mem_required = {}   
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
                    
                    num_weight = cw * k**2 * cout
                    self._w_mem_required[name] = num_weight * self.w_prec_eff[name] / 8
                    self._i_mem_required[name] = math.ceil(cin * i_prec / 8) * ih * iw * batch_size
                    self._o_mem_required[name] = math.ceil(cout * i_prec / 8) * oh * ow * batch_size
                else:
                    # input channel, output channel
                    cin, cout = w_dim
                    # batch size, sample size, output channel
                    batch_size, token_num, _ = o_dim

                    num_weight = cin * cout
                    self._w_mem_required[name] = num_weight * self.w_prec_eff[name] / 8
                    self._i_mem_required[name] = math.ceil(cin * i_prec / 8) * batch_size * token_num
                    if layer_idx == (len(self.layer_name_list) - 1):
                        self._o_mem_required[name] = 0
                    else:
                        self._o_mem_required[name] = math.ceil(cout * i_prec / 8) * batch_size * token_num

    def _get_quantized_weight(self, layer_name):
        for name, layer in self.model_q.named_modules():
            if ( layer_name == name ):
                w = layer.weight()
                wq = torch.int_repr(w)
                wqb_twosComplement = int_to_twosComplement(wq, w_bitwidth=8, device=self.DEVICE)
                if len(wqb_twosComplement.shape) == 5:
                    wqb_twosComplement = wqb_twosComplement.permute([0, 1, 3, 4, 2])
                return wqb_twosComplement
        raise Exception(f'ERROR! The layer {layer_name} cannot be found in the quantized model!')
    
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

