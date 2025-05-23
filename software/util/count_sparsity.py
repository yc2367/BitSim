import torch
import torch.nn as nn
from util.bin_int_convert import *


def count_zero_value_conv(wq_int):
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    param_count = K*C*W*H
    sparse_value_count = torch.sum(torch.eq(wq_int, 0))
    return int(sparse_value_count), int(param_count)


def count_zero_value_fc(wq_int):
    wq_int = wq_int.permute(1, 0)
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
    wq_int = wq_int.permute(1, 0)
    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    K, C = wq_int.size()
    param_count = K*C
    total_bit_count = w_bitwidth*param_count
    sparse_bit_count = total_bit_count - torch.sum(wqb_signMagnitude)
    return int(sparse_bit_count), int(total_bit_count)


def count_less_bit_sm_conv(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    print('testtt', wq_int.shape)
    K, C, H, W = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.reshape(NUM_GROUP, group_size)

    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    bit_one_count = torch.sum(wqb_signMagnitude, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
    sparse_bit_count = torch.sum(bit_one_count)
    total_bit_count = K * C * W * H * w_bitwidth

    return int(sparse_bit_count), int(total_bit_count)


def count_less_bit_sm_fc(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    wq_int = wq_int.permute(1, 0)
    K, C = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.unsqueeze(-1)
    wq_int = wq_int.reshape(NUM_GROUP, group_size)

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
    wq_int = wq_int.permute(1, 0)
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    K, C = wq_int.size()
    param_count = K*C
    total_bit_count = w_bitwidth*param_count
    sparse_bit_count = total_bit_count - torch.sum(wqb_twosComplement)
    return int(sparse_bit_count), int(total_bit_count)


def count_less_bit_2s_conv(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    K, C, H, W = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.reshape(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    bit_one_count = torch.sum(wqb_twosComplement, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
    sparse_bit_count = torch.sum(bit_one_count)
    total_bit_count = K * C * W * H * w_bitwidth

    return int(sparse_bit_count), int(total_bit_count)


def count_less_bit_2s_fc(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    wq_int = wq_int.permute(1, 0)
    K, C = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.unsqueeze(-1)
    wq_int = wq_int.reshape(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    bit_one_count = torch.sum(wqb_twosComplement, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
    sparse_bit_count = torch.sum(bit_one_count)
    total_bit_count = K * C * w_bitwidth

    return int(sparse_bit_count), int(total_bit_count)


def count_less_bit_clip_msb_conv(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.reshape(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)

    # tensor to store msb index of all groups
    msb_idx = torch.zeros([NUM_GROUP], device=device) 
    eq_msb_column = torch.ones([NUM_GROUP], device=device)
    for i in range(1, int(w_bitwidth)):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        msb_idx[eq_msb_column] += 1
    
    is_zero_column = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
    is_one_column = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(0, int(w_bitwidth)):
        is_zero_column[i] = torch.all(torch.eq(wqb_twosComplement[i], 0.), dim=-1)
        is_one_column[i] = torch.all(torch.eq(wqb_twosComplement[i], 1.), dim=-1)
    wqb_twosComplement[is_zero_column] = 0
    wqb_twosComplement[is_one_column] = 0
    
    for i in range(1, int(w_bitwidth)):
        set_column_to_zero = msb_idx.ge(i)
        wqb_twosComplement[i][set_column_to_zero] = 0
    
    bit_one_count = torch.sum(wqb_twosComplement, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
    sparse_bit_count = torch.sum(bit_one_count)
    total_bit_count = K * C * W * H * w_bitwidth
    return int(sparse_bit_count), int(total_bit_count)


def count_less_bit_clip_msb_fc(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    wq_int = wq_int.permute(1, 0)
    K, C = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.unsqueeze(-1)
    wq_int = wq_int.reshape(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)

    # tensor to store msb index of all groups
    msb_idx = torch.zeros([NUM_GROUP], device=device) 
    eq_msb_column = torch.ones([NUM_GROUP], device=device)
    for i in range(1, int(w_bitwidth)):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        msb_idx[eq_msb_column] += 1
    
    is_zero_column = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
    is_one_column = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(0, int(w_bitwidth)):
        is_zero_column[i] = torch.all(torch.eq(wqb_twosComplement[i], 0.), dim=-1)
        is_one_column[i] = torch.all(torch.eq(wqb_twosComplement[i], 1.), dim=-1)
    wqb_twosComplement[is_zero_column, :] = 0
    wqb_twosComplement[is_one_column, :] = 0
    
    for i in range(1, int(w_bitwidth)):
        set_column_to_zero = msb_idx.ge(i)
        wqb_twosComplement[i][set_column_to_zero] = 0
    bit_one_count = torch.sum(wqb_twosComplement, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
    sparse_bit_count = torch.sum(bit_one_count)
    total_bit_count = K * C * w_bitwidth
    return int(sparse_bit_count), int(total_bit_count)

def count_bbs_column_conv(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.reshape(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    
    is_zero_column = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
    is_one_column = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(0, int(w_bitwidth)):
        is_zero_column[i] = torch.all(torch.eq(wqb_twosComplement[i], 0.), dim=-1)
        is_one_column[i] = torch.all(torch.eq(wqb_twosComplement[i], 1.), dim=-1)

    return (torch.sum(is_zero_column).item() + torch.sum(is_one_column).item()) * group_size


def count_bbs_column_fc(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    wq_int = wq_int.permute(1, 0)
    K, C = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.unsqueeze(-1)
    wq_int = wq_int.reshape(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    
    is_zero_column = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
    is_one_column = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(0, int(w_bitwidth)):
        is_zero_column[i] = torch.all(torch.eq(wqb_twosComplement[i], 0.), dim=-1)
        is_one_column[i] = torch.all(torch.eq(wqb_twosComplement[i], 1.), dim=-1)
    return (torch.sum(is_zero_column).item() + torch.sum(is_one_column).item()) * group_size


class CountZeroColumn:
    def __init__(self) -> None:
        self.num_zero_column = 0
        self.num_zero_bits = 0
        self.num_total_column = 0
    
    def count_zero_column_conv(self, wq_int, w_bitwidth=8, group_size=16, device='cpu'):
        K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
        if C < group_size:
            group_size = C
        NUM_GROUP = K*W*H*C//group_size
        wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
        wq_int = wq_int.reshape(NUM_GROUP, group_size)

        wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)

        # check existing zero columns
        zero_column_mask = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
        for i in range(w_bitwidth):
            eq_zero = torch.all(torch.eq(wqb_signMagnitude[i], 0.), dim=-1)
            zero_column_mask[i][eq_zero] = True
        
        num_zero_column = torch.sum(zero_column_mask)
        self.num_zero_column += num_zero_column
        self.num_zero_bits += (num_zero_column * group_size)
        self.num_total_column += NUM_GROUP*w_bitwidth

        return (num_zero_column * group_size)

    def count_zero_column_fc(self, wq_int, w_bitwidth=8, group_size=16, device='cpu'):
        wq_int = wq_int.permute(1, 0)
        K, C = wq_int.size() # output channel, input channel, kernel width, kernel height
        if C < group_size:
            group_size = C
        NUM_GROUP = K*C//group_size
        wq_int = wq_int.unsqueeze(-1)
        wq_int = wq_int.reshape(NUM_GROUP, group_size)

        wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)

        # check existing zero columns
        zero_column_mask = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
        for i in range(w_bitwidth):
            eq_zero = torch.all(torch.eq(wqb_signMagnitude[i], 0.), dim=-1)
            zero_column_mask[i][eq_zero] = True
        
        num_zero_column = torch.sum(zero_column_mask)
        self.num_zero_column += num_zero_column
        self.num_zero_bits += (num_zero_column * group_size)
        self.num_total_column += NUM_GROUP*w_bitwidth

        return (num_zero_column * group_size)
