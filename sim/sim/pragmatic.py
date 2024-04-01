import math
import torch
import torch.nn as nn

from typing import List
from sim.stripes import Stripes
from sim.util.model_quantized import MODEL

from sim.util.bin_int_convert import int_to_twosComplement

# Pragmatic accelerator
class Pragmatic(Stripes):
    PR_SCALING = 1.3 # scaling factor to account for post placement and routing
    DISPATCHER_ENERGY_PER_COL = 0.072625 
    PE_ENERGY = 0.46625 * PR_SCALING # energy per 8-way DP PE, multiplied by 1.3 to account for post P&R
    #PE_ENERGY = 0.30625 * PR_SCALING
    W_REG_ENERGY_PER_ROW = 0.763 * PR_SCALING # energy (pJ) of the weight scheduler for a PE row
    #W_REG_ENERGY_PER_ROW = 0.38125 * PR_SCALING
    I_REG_ENERGY_PER_COL = (0.53 + DISPATCHER_ENERGY_PER_COL) * PR_SCALING # energy (pJ) of the activation register file for a PE column
    #I_REG_ENERGY_PER_COL = (0.2625 + DISPATCHER_ENERGY_PER_COL) * PR_SCALING
    PE_AREA = 1

    def __init__(self, 
                 input_precision_s: int, # bit-serial operand precision
                 input_precision_p: int, # bit-parallel operand precision
                 pe_dotprod_size: int, # length of the dot product inside one PE
                 pe_array_dim: List[int],
                 model_name: str,
                 model: nn.Module): # model comes from "BitSim/sim.model_profile/models/models.py
        self.model_q = MODEL[model_name].cpu() # quantized model
        super().__init__(input_precision_s, input_precision_p, pe_dotprod_size, 
                         pe_array_dim, model_name, model)
    
    def calc_cycle(self):
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
                        cycle_tile_k = torch.max(torch.sum(tile_k, dim=0))
                        cycle_kernel += int(cycle_tile_k.item())
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

                            cycle_tile_cw = torch.max(torch.sum(tile_cw, dim=0))
                            cycle_kernel += int(cycle_tile_cw.item())
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

                    cycle_tile_k = torch.max(torch.sum(tile_k, dim=0))
                    cycle_kernel += int(cycle_tile_k.item())
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
                
                cycle_tile_cin = torch.max(torch.sum(tile_cin, dim=0))
                cycle_kernel += int(cycle_tile_cin.item())
        cycle_batch = math.ceil(batch_size * sample_size / num_pe_col)

        total_cycle = cycle_kernel * cycle_batch
        return total_cycle
    
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
        
    def calc_compute_energy(self):
        num_pe = self.pe_array.total_unit_count
        num_cycle_compute = self.cycle_compute
        compute_energy = self.PE_ENERGY * num_pe * num_cycle_compute
        return compute_energy
    
    