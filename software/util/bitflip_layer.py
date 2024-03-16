import torch
import torch.nn as nn
from util.bin_int_convert import *


def bitflip_signMagnitude_conv(wq_int, w_bitwidth=8, group_size=16, zero_column_required=4, device='cpu'):
    K, C, W, H = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.view(K, W, H, C//group_size, group_size)

    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)

    # check existing zero columns
    zero_column_mask = torch.zeros([w_bitwidth, K, W, H, C//group_size], device=device)
    for i in range(int(w_bitwidth)):
        eq_zero = torch.eq(torch.sum(wqb_signMagnitude[i], dim=-1), 0.)
        zero_column_mask[i][eq_zero] = 1

    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = torch.full([K, W, H, C//group_size], w_bitwidth-zero_column_required, device=device)

    # cHeck if there are zero columns before prune_until
    for i in range(1, int(w_bitwidth)):
        mask = torch.logical_and(prune_until.gt(i), torch.eq(zero_column_mask[i], 1))
        prune_until[mask] += 1
    
    # test_until is a pointer to specify which column to test until
    test_until = torch.full([K, W, H, C//group_size], 1, device=device)
    for i in range(1, int(w_bitwidth)):
        mask = torch.logical_and(prune_until.gt(i), torch.eq(zero_column_mask[i], 1))
        test_until[mask] = i + 1
    
    # prune columns [prune_until:], we should test columns [test_until:] to minimize MSE
    # since the columns between test_until and prune_until can be adjusted arbitrarily as long as [prune_until:] are all zero
    value_test = torch.zeros_like(wq_int, dtype=torch.float32, device=device)
    for test_idx in range(1, int(w_bitwidth)):
        mask = torch.eq(test_until, test_idx)
        column_test = wqb_signMagnitude[test_idx:, mask, :]
        int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-test_idx, device=device)
        value_test[mask] = int_test
    
    value_new = torch.zeros_like(wq_int, dtype=torch.float32, device=device)
    for prune_idx in range(w_bitwidth-zero_column_required, w_bitwidth):
        for test_idx in range(1, prune_idx):
            mask_group = torch.logical_and(torch.eq(test_until, test_idx), torch.eq(prune_until, prune_idx))
            mask_group = mask_group.unsqueeze(-1).expand(-1, -1, -1, -1, group_size)
            error = torch.full(value_test.shape, 1e7, device=device)
            for n in range(2**(prune_idx-test_idx)):
                tmp_value = n * 2.**(w_bitwidth-prune_idx)
                new_error = (tmp_value - value_test) ** 2
                mask_value = torch.logical_and(torch.lt(new_error, error), mask_group)
                error[mask_value] = new_error[mask_value]
                value_new[mask_value] = tmp_value
            column_new = int_to_binary(value_new, w_bitwidth=w_bitwidth-test_idx, device=device)
            wqb_signMagnitude[test_idx:, mask_group] = column_new[:, mask_group]
    
    wq_int_new = signMagnitude_to_int(wqb_signMagnitude, w_bitwidth=w_bitwidth, device=device)
    wq_int_new = wq_int_new.view(K, W, H, C).permute(0, 3, 1, 2)

    return wq_int_new


def bitflip_signMagnitude_fc(wq_int, w_bitwidth=8, group_size=16, zero_column_required=4, device='cpu'):
    K, C = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    wq_int = wq_int.view(K, C//group_size, group_size)

    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)

    # check existing zero columns
    zero_column_mask = torch.zeros([w_bitwidth, K, C//group_size], device=device)
    for i in range(int(w_bitwidth)):
        eq_zero = torch.eq(torch.sum(wqb_signMagnitude[i], dim=-1), 0.)
        zero_column_mask[i][eq_zero] = 1

    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = torch.full([K, C//group_size], w_bitwidth-zero_column_required, device=device)

    # cHeck if there are zero columns before prune_until
    for i in range(1, int(w_bitwidth)):
        mask = torch.logical_and(prune_until.gt(i), torch.eq(zero_column_mask[i], 1))
        prune_until[mask] += 1
    
    # test_until is a pointer to specify which column to test until
    test_until = torch.full([K, C//group_size], 1, device=device)
    for i in range(1, int(w_bitwidth)):
        mask = torch.logical_and(prune_until.gt(i), torch.eq(zero_column_mask[i], 1))
        test_until[mask] = i + 1
    
    # prune columns [prune_until:], we should test columns [test_until:] to minimize MSE
    # since the columns between test_until and prune_until can be adjusted arbitrarily as long as [prune_until:] are all zero
    value_test = torch.zeros_like(wq_int, dtype=torch.float32, device=device)
    for test_idx in range(1, int(w_bitwidth)):
        mask = torch.eq(test_until, test_idx)
        column_test = wqb_signMagnitude[test_idx:, mask, :]
        int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-test_idx, device=device)
        value_test[mask] = int_test
    
    value_new = torch.zeros_like(wq_int, dtype=torch.float32, device=device)
    for prune_idx in range(w_bitwidth-zero_column_required, w_bitwidth):
        for test_idx in range(1, prune_idx):
            mask_group = torch.logical_and(torch.eq(test_until, test_idx), torch.eq(prune_until, prune_idx))
            mask_group = mask_group.unsqueeze(-1).expand(-1, -1, group_size)
            error = torch.full(value_test.shape, 1e7, device=device)
            for n in range(2**(prune_idx-test_idx)):
                tmp_value = n * 2.**(w_bitwidth-prune_idx)
                new_error = (tmp_value - value_test) ** 2
                mask_value = torch.logical_and(torch.lt(new_error, error), mask_group)
                error[mask_value] = new_error[mask_value]
                value_new[mask_value] = tmp_value
            column_new = int_to_binary(value_new, w_bitwidth=w_bitwidth-test_idx, device=device)
            wqb_signMagnitude[test_idx:, mask_group] = column_new[:, mask_group]
    
    wq_int_new = signMagnitude_to_int(wqb_signMagnitude, w_bitwidth=w_bitwidth, device=device)
    wq_int_new = wq_int_new.view(K, C)

    return wq_int_new


def bitflip_twosComplement_conv(wq_int, w_bitwidth=8, group_size=16, zero_column_required=4, device='cpu'):
    K, C, W, H = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.view(K, W, H, C//group_size, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)

    # tensor to store msb index of all groups
    msb_idx = torch.zeros([K, W, H, C//group_size], device=device) 
    eq_msb_column = torch.ones([K, W, H, C//group_size], device=device)
    for i in range(1, int(w_bitwidth)):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        msb_idx[eq_msb_column] += 1

    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = torch.full([K, W, H, C//group_size], w_bitwidth-zero_column_required, device=device)
    prune_until = prune_until + msb_idx
    
    column_test = torch.zeros_like(wqb_twosComplement, dtype=torch.float32, device=device)
    for prune_idx in range(w_bitwidth-zero_column_required, w_bitwidth):
        mask_group = torch.eq(prune_until, prune_idx)
        mask_value = mask_group.unsqueeze(-1).expand(-1, -1, -1, -1, group_size)
        column_test[prune_idx:, mask_value] = wqb_twosComplement[prune_idx:, mask_value]
        value_test = binary_to_int(column_test[prune_idx:], w_bitwidth=w_bitwidth-prune_idx, device=device)
        value_mean = torch.round(torch.mean(value_test, dim=-1))
        value_mean = value_mean.unsqueeze(-1).expand(-1, -1, -1, -1, group_size)
        column_new = int_to_binary(value_mean, w_bitwidth=w_bitwidth-prune_idx, device=device)
        wqb_twosComplement[prune_idx:, mask_value] = column_new[:, mask_value]

    wq_int_new = twosComplement_to_int(wqb_twosComplement, w_bitwidth=w_bitwidth, device=device)
    wq_int_new = wq_int_new.view(K, W, H, C).permute(0, 3, 1, 2)

    return wq_int_new


def bitflip_twosComplement_fc(wq_int, w_bitwidth=8, group_size=16, zero_column_required=4, device='cpu'):
    K, C = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    wq_int = wq_int.view(K, C//group_size, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)

    # tensor to store msb index of all groups
    msb_idx = torch.zeros([K, C//group_size], device=device) 
    eq_msb_column = torch.ones([K, C//group_size], device=device)
    for i in range(1, int(w_bitwidth)):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        msb_idx[eq_msb_column] += 1

    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = torch.full([K, C//group_size], w_bitwidth-zero_column_required, device=device)
    prune_until = prune_until + msb_idx
    
    column_test = torch.zeros_like(wqb_twosComplement, dtype=torch.float32, device=device)
    for prune_idx in range(w_bitwidth-zero_column_required, w_bitwidth):
        mask_group = torch.eq(prune_until, prune_idx)
        mask_value = mask_group.unsqueeze(-1).expand(-1, -1, group_size)
        column_test[prune_idx:, mask_value] = wqb_twosComplement[prune_idx:, mask_value]
        value_test = binary_to_int(column_test[prune_idx:], w_bitwidth=w_bitwidth-prune_idx, device=device)
        value_mean = torch.round(torch.mean(value_test, dim=-1))
        value_mean = value_mean.unsqueeze(-1).expand(-1,  -1, group_size)
        column_new = int_to_binary(value_mean, w_bitwidth=w_bitwidth-prune_idx, device=device)
        wqb_twosComplement[prune_idx:, mask_value] = column_new[:, mask_value]

    wq_int_new = twosComplement_to_int(wqb_twosComplement, w_bitwidth=w_bitwidth, device=device)
    wq_int_new = wq_int_new.view(K, C)

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


if __name__ == "__main__":
    t = torch.tensor([-15., -79.,  -5.,   2.,   0.,   9.,   0.,  -2.,  -4.,  -1.,  -8., -13.,
          3.,   3.,  -2.,   8.])
    tbs = int_to_signMagnitude(t, 8)
    tb2 = int_to_twosComplement(t, 8)
    a = search_zeroPoint(t, 8,4)
    b = bitFlip_signMagnitude(t, tbs, 8,4)
    c = bitFlip_twosComplement(t, tb2, 8,4)

    