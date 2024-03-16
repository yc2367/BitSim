import torch
import torch.nn as nn
from util.bitflip import *

def process_signMagnitude_conv(wq_int, w_bitwidth=8, group_size=16, zero_column_required=4, device='cpu'):
    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_signMagnitude = wqb_signMagnitude.to('cpu')
    wq_int_new = torch.zeros_like(wq_int)
    K, C, W, H = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for w in range(W):  # kernel width
            for h in range(H):  # kernel height
                for c in range(C // group_size):  # input channel 
                    group_q = wq_int[k, c*group_size:(c+1)*group_size, w, h]
                    group_qb = wqb_signMagnitude[:, k, c*group_size:(c+1)*group_size, w, h]
                    group_q_new = bitFlip_signMagnitude(group_q, group_qb, w_bitwidth=w_bitwidth,
                                                        zero_column_required=zero_column_required)
                    wq_int_new[k, c*group_size:(c+1)*group_size, w, h] = group_q_new
    return wq_int_new


def process_signMagnitude_fc(wq_int, w_bitwidth=8, group_size=16, zero_column_required=4, device='cpu'):
    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_signMagnitude = wqb_signMagnitude.to('cpu')
    wq_int_new = torch.zeros_like(wq_int)
    K, C = wq_int.shape # output channel, input channel
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for c in range(C // group_size):  # input channel 
            group_q = wq_int[k, c*group_size:(c+1)*group_size]
            group_qb = wqb_signMagnitude[:, k, c*group_size:(c+1)*group_size]
            group_q_new = bitFlip_signMagnitude(group_q, group_qb, w_bitwidth=w_bitwidth,
                                                zero_column_required=zero_column_required)
            wq_int_new[k, c*group_size:(c+1)*group_size] = group_q_new
    return wq_int_new


def process_zeroPoint_conv(wq_int, w_bitwidth=8, group_size=16, pruned_column_num=4, device='cpu'):
    wq_int_new = torch.zeros_like(wq_int)
    K, C, W, H = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for w in range(W):  # kernel width
            for h in range(H):  # kernel height
                for c in range(C // group_size):  # input channel 
                    group_q = wq_int[k, c*group_size:(c+1)*group_size, w, h]
                    group_q_new = search_zeroPoint(group_q, w_bitwidth=w_bitwidth, zero_column_required=pruned_column_num)
                    wq_int_new[k, c*group_size:(c+1)*group_size, w, h] = group_q_new
    return wq_int_new


def process_zeroPoint_fc(wq_int, w_bitwidth=8, group_size=16, pruned_column_num=4, device='cpu'):
    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_signMagnitude = wqb_signMagnitude.to('cpu')
    wq_int_new = torch.zeros_like(wq_int)
    K, C = wq_int.shape # output channel, input channel
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for c in range(C // group_size):  # input channel 
            group_q = wq_int[k, c*group_size:(c+1)*group_size]
            group_q_new = search_zeroPoint(group_q, w_bitwidth=w_bitwidth, zero_column_required=pruned_column_num)
            wq_int_new[k, c*group_size:(c+1)*group_size] = group_q_new
    return wq_int_new


def process_twosComplement_conv(wq_int, w_bitwidth=8, group_size=16, pruned_column_num=4, 
                                device='cpu', h_distance_target=0):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_twosComplement = wqb_twosComplement.to('cpu')
    wq_int_new = torch.zeros_like(wq_int)
    K, C, W, H = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for w in range(W):  # kernel width
            for h in range(H):  # kernel height
                for c in range(C // group_size):  # input channel 
                    group_q = wq_int[k, c*group_size:(c+1)*group_size, w, h]
                    group_qb = wqb_twosComplement[:, k, c*group_size:(c+1)*group_size, w, h]
                    group_q_new = bitFlip_twosComplement(group_q, group_qb, w_bitwidth=w_bitwidth,
                                                         zero_column_required=pruned_column_num, 
                                                         h_distance_target=h_distance_target)
                    wq_int_new[k, c*group_size:(c+1)*group_size, w, h] = group_q_new
    return wq_int_new


def process_twosComplement_fc(wq_int, w_bitwidth=8, group_size=16, pruned_column_num=4, 
                              device='cpu', h_distance_target=0):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_twosComplement = wqb_twosComplement.to('cpu')
    wq_int_new = torch.zeros_like(wq_int)
    K, C = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    for k in range(K):  # output channel
        for c in range(C // group_size):  # input channel 
            group_q = wq_int[k, c*group_size:(c+1)*group_size]
            group_qb = wqb_twosComplement[:, k, c*group_size:(c+1)*group_size]            
            group_q_new = bitFlip_twosComplement(group_q, group_qb, w_bitwidth=w_bitwidth,
                                                 zero_column_required=pruned_column_num,
                                                 h_distance_target=h_distance_target)
            wq_int_new[k, c*group_size:(c+1)*group_size] = group_q_new
    return wq_int_new


if __name__ == "__main__":
    t = torch.tensor([-15., -79.,  -5.,   2.,   0.,   9.,   0.,  -2.,  -4.,  -1.,  -8., -13.,
          3.,   3.,  -2.,   8.])
    tbs = int_to_signMagnitude(t, 8)
    tb2 = int_to_twosComplement(t, 8)
    a = search_zeroPoint(t, 8,4)
    b = bitFlip_signMagnitude(t, tbs, 8,4)
    c = bitFlip_twosComplement(t, tb2, 8,4)

    