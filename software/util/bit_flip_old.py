import torch
import torch.nn as nn
from util.bin_int_convert import *


def bitFlip_signMagnitude(group_q, group_qb, w_bitwidth=8, zero_column_required=4, return_binary=False):
    '''
    Apply bit-flip to a group of quantized weights in sign-magnitude format
    '''
    group_binary = group_qb.clone()
    # check existing zero columns
    zero_column_idx = [1e3]
    for i in range(1, int(w_bitwidth)):
        if torch.sum(group_qb[i]) == 0:
            zero_column_idx.append(i)
    
    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = w_bitwidth - zero_column_required 

    # cHeck if there are zero columns before prune_until
    for idx in zero_column_idx:
        if idx < prune_until:
            zero_column_required -= 1
            prune_until += 1
    
    # no need to prune
    if zero_column_required <= 0: 
        if return_binary:
            return group_binary
        else:
            return group_q
    
    # continue
    # test_until is a pointer to specify which column to test until
    zero_column_idx = [idx if idx < prune_until else 0 for idx in zero_column_idx]
    test_until = max(zero_column_idx) + 1
    
    # prune columns [prune_until:], we should test columns [test_until:] to minimize MSE
    # since the columns between test_until and prune_until can be adjusted arbitrarily as long as [prune_until:] are all zero
    column_test = group_binary[test_until:]
    int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-test_until)
    int_test_new = torch.zeros_like(int_test)

    for i, value in enumerate(int_test.tolist()):
        new_value = 0
        error = 1e7
        for n in range(2**(prune_until-test_until)):
            tmp_value = n * 2.**(w_bitwidth-prune_until)
            new_error = ((tmp_value - value)**2)
            if new_error < error:
                error = new_error
                new_value = tmp_value
            else:
                break

        int_test_new[i] = new_value

    column_test_new = int_to_binary(int_test_new, w_bitwidth=w_bitwidth-test_until)
    group_binary[test_until:] = column_test_new
    group_int_new = signMagnitude_to_int(group_binary, w_bitwidth=w_bitwidth)

    if return_binary:
        return group_binary
    else:
        return group_int_new
    

def bitFlip_twosComplement(group_q, group_qb, w_bitwidth=8, zero_column_required=4, 
                           return_binary=False, h_distance_target=0):
    '''
    Apply bit-flip to a group of quantized weights in 2's complement format
    '''
    group_binary = group_qb.clone()
    # check existing zero columns
    zero_column_idx = [1e3]
    for i in range(1, int(w_bitwidth)):
        if torch.equal(group_binary[i], group_binary[0]):
            zero_column_idx.append(i)
        else:
            break
    
    '''
        tmp_idx: the next column after the sign column 
        if the hamming distance between column tmp_idx and column 0 is small 
        we can use a special unit to calculate column tmp_idx 
    '''
    if (zero_column_idx[-1] < w_bitwidth):
        tmp_idx = zero_column_idx[-1] + 1
    else:
        tmp_idx = 1
    if tmp_idx < w_bitwidth:
        hamming_distance = torch.sum(torch.abs(group_binary[tmp_idx] - group_binary[0]))
        if hamming_distance < h_distance_target:
            zero_column_idx.append(tmp_idx)

    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = w_bitwidth - zero_column_required 

    # cHeck if there are zero columns before prune_until
    for idx in zero_column_idx:
        if idx < prune_until:
            zero_column_required -= 1
            prune_until += 1
    
    # no need to prune
    if zero_column_required <= 0: 
        if return_binary:
            return group_binary
        else:
            return group_q

    # prune columns [prune_until:], we should test columns [prune_until:] to minimize MSE
    # since the value should be the same for all weight columns [prune_until:] in a group, the value can be adjusted arbitrarily
    if prune_until == 1:
        int_test = twosComplement_to_int(group_binary, w_bitwidth=w_bitwidth)
        value_mean = torch.round(torch.mean(int_test))
        int_test_new = torch.Tensor([value_mean for _ in range(group_q.shape[0])])
        group_int_new = int_test_new
    else:
        column_test = group_binary[prune_until:]
        int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-prune_until)
        value_mean = torch.round(torch.mean(int_test))
        int_test_new = torch.Tensor([value_mean for _ in range(group_q.shape[0])])
        column_test_new = int_to_binary(int_test_new, w_bitwidth=w_bitwidth-prune_until)
        group_binary[prune_until:] = column_test_new
        group_int_new = twosComplement_to_int(group_binary, w_bitwidth=w_bitwidth)

    if return_binary:
        return group_binary
    else:
        return group_int_new
    

def bitFlip_signMagnitude_zeroPoint(group_q, group_qb, w_bitwidth=8, zero_column_required=4, return_binary=False):
    '''
    Apply bit-flip to a group of quantized weights in sign-magnitude format
    '''
    group_binary = group_qb.clone()
    # check existing zero columns
    zero_column_idx = [1e3]
    for i in range(1, int(w_bitwidth)):
        if torch.sum(group_qb[i]) == 0:
            zero_column_idx.append(i)
            zero_column_required -= 1
        else:
            break
    
    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = w_bitwidth - zero_column_required 
    
    # no need to prune
    if zero_column_required <= 0: 
        if return_binary:
            return group_binary
        else:
            return group_q
    
    # continue
    # test_until is a pointer to specify which column to test until
    zero_column_idx = [idx if idx < prune_until else 0 for idx in zero_column_idx]
    test_until = max(zero_column_idx) + 1
    
    # prune columns [prune_until:], we should test columns [test_until:] to minimize MSE
    # since the columns between test_until and prune_until can be adjusted arbitrarily as long as [prune_until:] are all zero
    column_test = group_binary[test_until:]
    int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-test_until)
    int_test_new = torch.zeros_like(int_test)

    for i, value in enumerate(int_test.tolist()):
        new_value = 0
        error = 1e7
        for n in range(2**(prune_until-test_until)):
            tmp_value = n * 2.**(w_bitwidth-prune_until)
            new_error = ((tmp_value - value)**2)
            if new_error < error:
                error = new_error
                new_value = tmp_value
            else:
                break

        int_test_new[i] = new_value

    column_test_new = int_to_binary(int_test_new, w_bitwidth=w_bitwidth-test_until)
    group_binary[test_until:] = column_test_new
    group_int_new = signMagnitude_to_int(group_binary, w_bitwidth=w_bitwidth)

    if return_binary:
        return group_binary
    else:
        return group_int_new
    

def search_zeroPoint(group_q, w_bitwidth=8, zero_column_required=4):
    # check existing zero columns
    v_max = 2.**(w_bitwidth-1) - 1
    v_min = -v_max
    group_qf = group_q.to(torch.float32)
    error = 1e8
    group_int_with_offset = torch.zeros_like(group_qf)
    for o in range(-16, 15):
        offset = float(o)
        group_q_new = torch.Tensor(group_qf + offset)
        group_q_new[group_q_new.lt(v_min)] = v_min
        group_q_new[group_q_new.gt(v_max)] = v_max
        group_qb_new = int_to_signMagnitude(group_q_new, w_bitwidth=w_bitwidth)
        group_int = bitFlip_signMagnitude_zeroPoint(group_q_new, group_qb_new, w_bitwidth, 
                                                    zero_column_required)
        group_int = group_int - offset 
        new_error = torch.mean( (group_int - group_qf)**2 )
        if new_error < error:
            error = new_error
            group_int_with_offset = group_int

    return group_int_with_offset

