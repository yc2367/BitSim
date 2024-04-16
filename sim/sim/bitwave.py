import math, time
import torch
import torch.nn as nn

from typing import List, Dict
from sim.stripes import Stripes
from hw.mem.mem_instance import MemoryInstance
from hw.alu.alu_unit import BitSerialPE

from sim.util.bin_int_convert import int_to_signMagnitude

# Bitwave accelerator
class Bitwave(Stripes):
    PR_SCALING = 1.5 # scaling factor to account for post placement and routing
    RECONFIG_SCALING = 1.1 # scaling factor to account for BitWave's reconfigurable dataflow
    DISPATCHER_ENERGY_PER_COL = 0.072625 
    PE_ENERGY = 0.2475 * PR_SCALING * RECONFIG_SCALING # energy per 8-way DP PE
    #PE_ENERGY = 0.30625 * PR_SCALING
    W_SCHEDULER_ENERGY_PER_ROW = 0.06575 * PR_SCALING # energy (pJ) of the weight scheduler for a PE row
    PE_AREA = 1

    def __init__(self, 
                 input_precision_s: int, # bit-serial operand precision
                 input_precision_p: int, # bit-parallel operand precision
                 pe_dotprod_size: int, # length of the dot product inside one PE
                 pe_array_dim: List[int],
                 model_name: str,
                 layer_prec: Dict={},    # the precision of every layer
                 en_bitflip: bool=False  # whether enable bitflip
                 ): 
        super().__init__(input_precision_s, input_precision_p, pe_dotprod_size, 
                         pe_array_dim, model_name, init_mem=False)
        self.model_q = self._get_quantized_model() # quantized model

        self.en_bitflip = en_bitflip
        # supported dataflow in BitWave
        # every tuple indicates (pe_dotprod_size, pe_array_height, pe_array_width)
        self.dataflow = [(8, 32, 16), (16, 32, 8), (32, 32, 4), (128, 8, 1), 
                         (16, 64, 1), (32, 32, 1), (16, 1, 16)]
        # to be modified later
        self.w_prec_config = {}
        for name in self.layer_name_list:
            self.w_prec_config[name] = input_precision_s

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
        self._layer_dataflow = {}
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            cycle_layer_compute = 1e9
            best_dataflow = None
            if w_dim is not None:
                for dataflow in self.dataflow:
                    if len(w_dim) == 4:
                        cin = i_dim[3]
                        cw  = w_dim[2]
                        if cin == cw: 
                            cycle_layer_compute_new = self._calc_cycle_conv2d(name, w_dim, o_dim, dataflow)
                        else: # depthwise conv
                            cycle_layer_compute_new = self._calc_cycle_dwconv(name, w_dim, i_dim, o_dim, dataflow)
                    else:
                        cycle_layer_compute_new = self._calc_cycle_fc(name, w_dim, o_dim, dataflow)
                    if cycle_layer_compute_new < cycle_layer_compute:
                        cycle_layer_compute = cycle_layer_compute_new
                        best_dataflow = dataflow
                self._layer_cycle_compute[name] = cycle_layer_compute
                self._layer_dataflow[name] = best_dataflow
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
    
    def _calc_cycle_conv2d(self, layer_name, w_dim, o_dim, dataflow):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]
        wq_b = wq_b[1:, :, :, :, :]

        pe_group_size = dataflow[0]
        num_pe_row = dataflow[1]
        num_pe_col = dataflow[2]

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
                            cycle_tile_k = self.w_prec_config[layer_name] - 1

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
                                cycle_tile_cw = self.w_prec_config[layer_name] - 1

                            cycle_kernel += int(cycle_tile_cw)
        cycle_ow = math.ceil(ow / num_pe_col)
        cycle_oh = oh

        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch 
        return total_cycle
    
    def _calc_cycle_dwconv(self, layer_name, w_dim, i_dim, o_dim, dataflow):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]
        wq_b = wq_b[1:, :, :, :, :]

        pe_group_size = dataflow[0]
        num_pe_col = dataflow[2]

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
                        cycle_tile_k = self.w_prec_config[layer_name] - 1

                    cycle_kernel += int(cycle_tile_k)
        cycle_ow = math.ceil(ow / num_pe_col)
        cycle_oh = oh

        cycle_per_batch = (cycle_kernel * cycle_ow * cycle_oh)
        total_cycle = cycle_per_batch 
        return total_cycle

    def _calc_cycle_fc(self, layer_name, w_dim, o_dim, dataflow):
        # wq_b dimension: [bit_significance, cout, k, k, cw]
        wq_b = self._get_quantized_weight(layer_name) # [bit_significance]
        wq_b = wq_b[1:, :, :]

        pe_group_size = dataflow[0]
        num_pe_row = dataflow[1]
        num_pe_col = dataflow[2]

        # input channel, output channel
        cin, cout = w_dim
        # batch size, sample size, output channel
        batch_size, token_num, _ = o_dim

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
                    cycle_tile_cin = self.w_prec_config[layer_name] - 1

                cycle_kernel += int(cycle_tile_cin)
        cycle_batch = math.ceil(token_num / num_pe_col)

        total_cycle = cycle_kernel * cycle_batch
        return total_cycle
    
    def calc_pe_array_tile(self):
        if not bool(self.dataflow):
            self.calc_cycle()
        total_tile = 0
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            dataflow = self._layer_dataflow[name]
            if w_dim is not None:
                if len(w_dim) == 4:
                    cin = i_dim[3]
                    cw  = w_dim[2]
                    if cin == cw: 
                        total_tile += self._calc_tile_conv2d(w_dim, o_dim, dataflow)
                    else: # depthwise conv
                        total_tile += self._calc_tile_dwconv(w_dim, i_dim, o_dim, dataflow)
                else:
                    total_tile += self._calc_tile_fc(w_dim, o_dim, dataflow)
        return total_tile
    
    def _calc_tile_conv2d(self, w_dim, o_dim, dataflow):
        pe_group_size = dataflow[0]
        num_pe_row = dataflow[1]
        num_pe_col = dataflow[2]

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
    
    def _calc_tile_dwconv(self, w_dim, i_dim, o_dim, dataflow):
        pe_group_size = dataflow[0]
        num_pe_col = dataflow[2]

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

    def _calc_tile_fc(self, w_dim, o_dim, dataflow):
        pe_group_size = dataflow[0]
        num_pe_row = dataflow[1]
        num_pe_col = dataflow[2]

        # input channel, output channel
        cin, cout = w_dim
        # batch size, sample size, output channel
        batch_size, token_num, _ = o_dim

        # tile_in_channel:   number of tiles along input channel
        # tile_cout:  number of tiles along output channel
        tile_in_channel  = math.ceil(cin / pe_group_size)
        tile_cout        = math.ceil(cout / num_pe_row)
        tile_batch       = math.ceil(token_num / num_pe_col)

        total_tile = (tile_in_channel * tile_cout * tile_batch)
        return total_tile
    
    def _calc_dram_cycle(self):
        self._layer_cycle_dram = {}
        i_prec = self.pe.input_precision_p
        dram_bandwidth = self.dram.rw_bw * 2 # DDR
        size_sram_i = self.i_sram.size / 8
        self._read_layer_input = True

        for name in self.layer_name_list:
            w_prec = self.w_prec_config[name]
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
    
    def calc_compute_energy(self):
        num_pe = self.total_pe_count
        if self.cycle_compute is None:
            self.cycle_compute, _ = self.calc_cycle()
        num_cycle_compute = self.cycle_compute
        compute_energy = self.PE_ENERGY * num_pe * num_cycle_compute
        # Bitwave has 128 weight schedulers to account for cout = 128 in the flexible dataflow
        compute_energy += (self.W_SCHEDULER_ENERGY_PER_ROW * 128 * num_cycle_compute)
        return compute_energy
    
    def calc_sram_rd_energy(self):
        w_sram_rd_cost = self.w_sram.r_cost
        i_sram_rd_cost = self.i_sram.r_cost
        if self.cycle_compute is None:
            self.cycle_compute, _ = self.calc_cycle()
        num_cycle_compute = self.cycle_compute
        num_tile = self.calc_pe_array_tile()

        w_sram_rd_energy = num_cycle_compute * w_sram_rd_cost
        i_sram_rd_energy = num_tile * i_sram_rd_cost
        total_energy = w_sram_rd_energy + i_sram_rd_energy
        return total_energy
    
    def calc_sram_wr_energy(self):
        total_energy = 0
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            w_prec = self.w_prec_config[name]
            i_prec = self.pe.input_precision_p
            if w_dim is not None:
                if len(w_dim) == 4:
                    total_energy += self._calc_sram_wr_energy_conv(name, w_dim, i_dim, o_dim, 
                                                                   w_prec, i_prec)
                else:
                    total_energy += self._calc_sram_wr_energy_fc(name, w_dim, i_dim, o_dim, 
                                                                 w_prec, i_prec)
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
    
    def _check_layer_mem_size(self):
        self._w_mem_required = {}
        self._i_mem_required = {}
        self._o_mem_required = {}   
        i_prec = self.pe.input_precision_p
        for layer_idx, name in enumerate(self.layer_name_list):
            w_prec = self.w_prec_config[name]
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
                    
                    self._w_mem_required[name] = math.ceil(cw / 8) * w_prec * k**2 * cout
                    self._i_mem_required[name] = math.ceil(cin * i_prec / 8) * ih * iw 
                    self._o_mem_required[name] = math.ceil(cout * i_prec / 8) * oh * ow 
                else:
                    # input channel, output channel
                    cin, cout = w_dim
                    # batch size, sample size, output channel
                    batch_size, token_num, _ = o_dim

                    self._w_mem_required[name] = math.ceil(cin / 8) * w_prec * cout
                    self._i_mem_required[name] = math.ceil(cin * i_prec / 8)  * token_num
                    if layer_idx == (len(self.layer_name_list) - 1):
                        self._o_mem_required[name] = 0
                    else:
                        self._o_mem_required[name] = math.ceil(cout * i_prec / 8)  * token_num
                        
    def _get_quantized_weight(self, layer_name):
        for name, wq in self.model_q.items():
            if ( layer_name == name ):
                wqb_signMagnitude = int_to_signMagnitude(wq, w_bitwidth=8, device=self.DEVICE)
                if len(wqb_signMagnitude.shape) == 5:
                    wqb_signMagnitude = wqb_signMagnitude.permute([0, 1, 3, 4, 2])
                return wqb_signMagnitude
        raise Exception(f'ERROR! The layer {layer_name} cannot be found in the quantized model!')
    
    def _init_mem(self):
        w_sram_bank = 16 # one bank feeds 2 PE rows
        w_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 18 * 1024*8 * w_sram_bank, 
                            'bank_count': w_sram_bank, 
                            'rw_bw': 64 * w_sram_bank, 
                            'r_port': 1, 
                            'w_port': 1, 
                            'rw_port': 0,
                        }
        self.w_sram = MemoryInstance('w_sram', w_sram_config, 
                                     r_cost=0, w_cost=0, latency=1, area=0, 
                                     min_r_granularity=None, 
                                     min_w_granularity=64, 
                                     get_cost_from_cacti=True, double_buffering_support=False)
        
        i_sram_bank = 16 # one bank feeds 1 PE columns
        i_sram_config = {
                            'technology': 0.028,
                            'mem_type': 'sram', 
                            'size': 16 * 1024*8 * i_sram_bank, 
                            'bank_count': i_sram_bank, 
                            'rw_bw': 64 * i_sram_bank,
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
