import torch
import torch.nn as nn
from util.bin_int_convert import *


def count_zero_bit_conv(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_twosComplement = wqb_twosComplement.to(device)
    sparse_bit_count = 0
    K, C, W, H = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for w in range(W):  # kernel width
            for h in range(H):  # kernel height
                for c in range(C // group_size):  # input channel 
                    group_qb = wqb_twosComplement[:, k, c*group_size:(c+1)*group_size, w, h]
                    sparse_bit_count += (w_bitwidth*group_size - torch.sum(group_qb))
    total_bit_count = K * C * W * H * w_bitwidth
    return sparse_bit_count, total_bit_count


def count_zero_bit_fc(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_twosComplement = wqb_twosComplement.to(device)
    sparse_bit_count = 0
    K, C = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for c in range(C // group_size):  # input channel 
            group_qb = wqb_twosComplement[:, k, c*group_size:(c+1)*group_size]            
            sparse_bit_count += (w_bitwidth*group_size - torch.sum(group_qb))
    total_bit_count = K * C * w_bitwidth
    return sparse_bit_count, total_bit_count


def count_less_bit_conv(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_twosComplement = wqb_twosComplement.to(device)
    sparse_bit_count = 0
    K, C, W, H = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for w in range(W):  # kernel width
            for h in range(H):  # kernel height
                for c in range(C // group_size):  # input channel 
                    group_qb = wqb_twosComplement[:, k, c*group_size:(c+1)*group_size, w, h]
                    bit_one_count = torch.sum(group_qb, dim=1)
                    #print('a', bit_one_count)
                    skip_zero = bit_one_count.lt(group_size/2)
                    #print('b', skip_zero)
                    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
                    #print('c', bit_one_count)
                    sparse_bit_count += torch.sum(bit_one_count)
    total_bit_count = K * C * W * H * w_bitwidth
    return sparse_bit_count, total_bit_count


def count_less_bit_fc(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_twosComplement = wqb_twosComplement.to(device)
    sparse_bit_count = 0
    K, C = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for c in range(C // group_size):  # input channel 
            group_qb = wqb_twosComplement[:, k, c*group_size:(c+1)*group_size] 
            bit_one_count = torch.sum(group_qb, dim=1)
            skip_zero = bit_one_count.lt(group_size/2)
            bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
            sparse_bit_count += torch.sum(bit_one_count)
    total_bit_count = K * C * w_bitwidth
    return sparse_bit_count, total_bit_count


def count_less_bit_clip_msb_conv(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_twosComplement = wqb_twosComplement.to(device)
    sparse_bit_count = 0
    K, C, W, H = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for w in range(W):  # kernel width
            for h in range(H):  # kernel height
                for c in range(C // group_size):  # input channel 
                    group_qb = wqb_twosComplement[:, k, c*group_size:(c+1)*group_size, w, h]
                    msb_idx = 0
                    for i in range(1, int(w_bitwidth)):
                        if torch.equal(group_qb[0, :], group_qb[i, :]):
                            msb_idx += 1
                        else:
                            break
                    bit_one_count = torch.sum(group_qb[msb_idx:int(w_bitwidth), :], dim=1)
                    skip_zero = bit_one_count.lt(group_size/2)
                    bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
                    sparse_bit_count += (torch.sum(bit_one_count) + group_size*msb_idx)
    total_bit_count = K * C * W * H * w_bitwidth
    return sparse_bit_count, total_bit_count


def count_less_bit_clip_msb_fc(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_twosComplement = wqb_twosComplement.to(device)
    sparse_bit_count = 0
    K, C = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for c in range(C // group_size):  # input channel 
            group_qb = wqb_twosComplement[:, k, c*group_size:(c+1)*group_size] 
            msb_idx = 0
            for i in range(1, int(w_bitwidth)):
                if torch.equal(group_qb[0, :], group_qb[i, :]):
                    msb_idx += 1
                else:
                    break
            bit_one_count = torch.sum(group_qb[msb_idx:int(w_bitwidth), :], dim=1)
            skip_zero = bit_one_count.lt(group_size/2)
            bit_one_count[skip_zero] = group_size - bit_one_count[skip_zero]
            sparse_bit_count += (torch.sum(bit_one_count) + group_size*msb_idx)
    total_bit_count = K * C * w_bitwidth
    return sparse_bit_count, total_bit_count

