import torch
import torch.nn as nn
from sim.util.bin_int_convert import *


def count_zero_value_conv(wq_int):
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    param_count = K*C*W*H
    sparse_value_count = torch.sum(torch.eq(wq_int, 0))
    return int(sparse_value_count), int(param_count)


def count_zero_value_fc(wq_int):
    K, C = wq_int.size()
    param_count = K*C
    sparse_value_count = torch.sum(torch.eq(wq_int, 0))
    return int(sparse_value_count), int(param_count)


def count_zero_bit_sm_conv(wq_int, w_bitwidth=8, device='cpu'):
    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    param_count = K*C*W*H
    total_bit_count = w_bitwidth*param_count
    sparse_bit_count = total_bit_count - torch.sum(wqb_signMagnitude)
    return int(sparse_bit_count), int(total_bit_count)


def count_zero_bit_sm_fc(wq_int, w_bitwidth=8, device='cpu'):
    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    K, C = wq_int.size()
    param_count = K*C
    total_bit_count = w_bitwidth*param_count
    sparse_bit_count = total_bit_count - torch.sum(wqb_signMagnitude)
    return int(sparse_bit_count), int(total_bit_count)


def count_less_bit_sm_conv(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    K, C, H, W = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    bit_one_count = torch.sum(wqb_signMagnitude, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
    sparse_bit_count = torch.sum(bit_one_count)
    total_bit_count = K * C * W * H * w_bitwidth

    return int(sparse_bit_count), int(total_bit_count)


def count_less_bit_sm_fc(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    K, C = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    bit_one_count = torch.sum(wqb_signMagnitude, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
    sparse_bit_count = torch.sum(bit_one_count)
    total_bit_count = K * C * w_bitwidth

    return int(sparse_bit_count), int(total_bit_count)


def count_zero_bit_2s_conv(wq_int, w_bitwidth=8, device='cpu'):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    param_count = K*C*W*H
    total_bit_count = w_bitwidth*param_count
    sparse_bit_count = total_bit_count - torch.sum(wqb_twosComplement)
    return int(sparse_bit_count), int(total_bit_count)


def count_zero_bit_2s_fc(wq_int, w_bitwidth=8, device='cpu'):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    K, C = wq_int.size()
    param_count = K*C
    total_bit_count = w_bitwidth*param_count
    sparse_bit_count = total_bit_count - torch.sum(wqb_twosComplement)
    return int(sparse_bit_count), int(total_bit_count)


def count_less_bit_2sComplement_old(wq_b, w_bitwidth=8, group_size=16, num_pruned_column=2):
    eq_msb_column = torch.ones(wq_b.shape[1], device=wq_b.device)
    for i in range(1, w_bitwidth-num_pruned_column):
        eq_column = torch.all(torch.eq(wq_b[0], wq_b[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        wq_b[i, eq_msb_column, :] = 0

    bit_one_count = torch.sum(wq_b, dim=-1)
    is_skip_one = bit_one_count.gt(group_size/2)
    bit_one_count[is_skip_one] = group_size - bit_one_count[is_skip_one]

    return bit_one_count

def count_less_bit_2sComplement(wq_b, w_bitwidth=8, group_size=16, num_pruned_column=2):
    wb = wq_b.clone()
    eq_msb_column = torch.ones(wb.shape[1], device=wb.device)
    for i in range(1, num_pruned_column + 1):
        eq_column = torch.all(torch.eq(wb[0], wb[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        wb[i, eq_msb_column, :] = 0
    
    is_redun_col = torch.logical_or(torch.all(wb.logical_not(), dim=-1), torch.all(wb, dim=-1))
    num_redun_col_per_group = torch.sum(is_redun_col, dim=0)
    has_more_redun_col = num_redun_col_per_group.gt(num_pruned_column)
    redun_load = torch.sum(num_redun_col_per_group[has_more_redun_col] - num_pruned_column)
    #print(redun_load)
    bit_one_count = torch.sum(wb, dim=-1)
    mask_skip = bit_one_count.lt(group_size/2)
    wb[mask_skip] = 1 - wb[mask_skip]

    return wb, redun_load


def count_less_bit_clip_msb_conv(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)

    # tensor to store msb index of all groups
    msb_idx = torch.zeros([NUM_GROUP], device=device) 
    eq_msb_column = torch.ones([NUM_GROUP], device=device)
    for i in range(1, int(w_bitwidth)):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        msb_idx[eq_msb_column] += 1
    
    for i in range(1, int(w_bitwidth)):
        set_column_to_zero = msb_idx.ge(i)
        wqb_twosComplement[i][set_column_to_zero] = wqb_twosComplement[i][set_column_to_zero] * 0
    bit_one_count = torch.sum(wqb_twosComplement, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
    sparse_bit_count = torch.sum(bit_one_count)
    total_bit_count = K * C * W * H * w_bitwidth
    return int(sparse_bit_count), int(total_bit_count)


def count_less_bit_clip_msb_fc(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    K, C = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)

    # tensor to store msb index of all groups
    msb_idx = torch.zeros([NUM_GROUP], device=device) 
    eq_msb_column = torch.ones([NUM_GROUP], device=device)
    for i in range(1, int(w_bitwidth)):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        msb_idx[eq_msb_column] += 1
    
    for i in range(1, int(w_bitwidth)):
        set_column_to_zero = msb_idx.ge(i)
        wqb_twosComplement[i][set_column_to_zero] = wqb_twosComplement[i][set_column_to_zero] * 0
    bit_one_count = torch.sum(wqb_twosComplement, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
    sparse_bit_count = torch.sum(bit_one_count)
    total_bit_count = K * C * w_bitwidth
    return int(sparse_bit_count), int(total_bit_count)

class countZeroColumn:
    def __init__(self) -> None:
        self.num_zero_column = 0
        self.num_total_column = 0
    
    def count_zero_column_conv(self, wq_int, w_bitwidth, group_size, device='cpu'):
        K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
        if C < group_size:
            group_size = C
        NUM_GROUP = K*W*H*C//group_size
        wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
        wq_int = wq_int.view(NUM_GROUP, group_size)

        wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)

        # check existing zero columns
        zero_column_mask = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
        for i in range(w_bitwidth):
            eq_zero = torch.all(torch.eq(wqb_signMagnitude[i], 0.), dim=-1)
            zero_column_mask[i][eq_zero] = True
        
        self.num_zero_column += torch.sum(zero_column_mask)
        self.num_total_column += NUM_GROUP*w_bitwidth

    def count_zero_column_fc(self, wq_int, w_bitwidth, group_size, device='cpu'):
        K, C = wq_int.size() # output channel, input channel, kernel width, kernel height
        if C < group_size:
            group_size = C
        NUM_GROUP = K*C//group_size
        wq_int = wq_int.unsqueeze(-1)
        wq_int = wq_int.view(NUM_GROUP, group_size)

        wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)

        # check existing zero columns
        zero_column_mask = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
        for i in range(w_bitwidth):
            eq_zero = torch.all(torch.eq(wqb_signMagnitude[i], 0.), dim=-1)
            zero_column_mask[i][eq_zero] = True
        
        self.num_zero_column += torch.sum(zero_column_mask)
        self.num_total_column += NUM_GROUP*w_bitwidth
