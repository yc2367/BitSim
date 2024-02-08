import torch

def int_to_signMagnitude(weight_q, w_bitwidth=8, cellBit=1, device='cpu'):
    weight_q = weight_q.to(device, copy=True)
    cellRange = 2**cellBit
    is_min = weight_q.eq(-2**(w_bitwidth-1))
    weight_q[is_min] = -2**(w_bitwidth-1) + 1
    is_neg = weight_q.lt(0.)
    weight_q[is_neg] = -weight_q[is_neg]
    
    weight_q_shape = torch.Tensor(list(weight_q.size()))
    bin_list_shape = torch.cat((torch.Tensor([w_bitwidth/cellBit]), weight_q_shape)).to(int).tolist()
    remainder_list = torch.zeros(bin_list_shape, device=device).type_as(weight_q)

    for k in reversed(range(int(w_bitwidth/cellBit))):
        remainder = torch.fmod(weight_q, cellRange)
        remainder_list[k] = remainder
        weight_q = torch.round((weight_q-remainder)/cellRange)

    remainder_list[0, is_neg] = 1.
    return remainder_list


def signMagnitude_to_int(wqb_list, w_bitwidth=8):
    bin_list_shape = wqb_list.size()
    wq_list_shape = list(bin_list_shape[1:])
    wq_list = torch.zeros(wq_list_shape)

    for k in reversed(range(int(w_bitwidth))):
        if k != 0:
          wq_list += (wqb_list[k] * 2.**(w_bitwidth-1-k))
        else:
          ones  = wqb_list[k].eq(1.)
          wq_list[ones] = -wq_list[ones]
    return wq_list


def int_to_twosComplement(weight_q, w_bitwidth=8, cellBit=1, device='cpu'):
    weight_q = weight_q.to(device, copy=True)

    cellRange = 2**cellBit
    is_neg = weight_q.lt(0.)
    weight_q[is_neg] = 2**(w_bitwidth-1) + weight_q[is_neg]
    
    weight_q_shape = torch.Tensor(list(weight_q.size()))
    bin_list_shape = torch.cat((torch.Tensor([w_bitwidth/cellBit]), weight_q_shape)).to(int).tolist()
    remainder_list = torch.zeros(bin_list_shape, device=device).type_as(weight_q)

    for k in reversed(range(int(w_bitwidth/cellBit))):
        remainder = torch.fmod(weight_q, cellRange)
        remainder_list[k] = remainder
        weight_q = torch.round((weight_q-remainder)/cellRange)
    
    remainder_list[0, is_neg] = 1.
    return remainder_list


def twosComplement_to_int(wqb_list, w_bitwidth=8):
    bin_list_shape = wqb_list.size()
    wq_list_shape = list(bin_list_shape[1:])
    wq_list = torch.zeros(wq_list_shape)

    for k in reversed(range(int(w_bitwidth))):
        if k != 0:
          wq_list += (wqb_list[k] * 2.**(w_bitwidth-1-k))
        else:
          wq_list -= (wqb_list[k] * 2.**(w_bitwidth-1-k))

    return wq_list


def int_to_binary(weight_q, w_bitwidth=8, cellBit=1):
    weight_q = weight_q.clone()

    cellRange = 2**cellBit
    weight_q_shape = torch.Tensor(list(weight_q.size()))
    bin_list_shape = torch.cat((torch.Tensor([w_bitwidth/cellBit]), weight_q_shape)).to(int).tolist()
    remainder_list = torch.zeros(bin_list_shape).type_as(weight_q)

    for k in reversed(range(int(w_bitwidth/cellBit))):
        remainder = torch.fmod(weight_q, cellRange)
        remainder_list[k] = remainder
        weight_q = torch.round((weight_q-remainder)/cellRange)
    
    return remainder_list


def binary_to_int(wqb_list, w_bitwidth=8):
    bin_list_shape = wqb_list.size()
    wq_list_shape = list(bin_list_shape[1:])
    wq_list = torch.zeros(wq_list_shape)

    for k in reversed(range(int(w_bitwidth))):
        wq_list += (wqb_list[k] * 2.**(w_bitwidth-1-k))
    
    return wq_list


def take_twosComplement(wqb_list, w_bitwidth=8, cellbit=1):
    '''
    Take 2's complement of a number. 
    E.g.,   81  = b'01010001
    return, -81 = b'10101111
    '''
    wqb_list = wqb_list.clone()
    new_wqb_list = torch.zeros_like(wqb_list)

    ones  = wqb_list.eq(1.)
    zeros = wqb_list.eq(0.)

    # invert the bits for adding 1
    wqb_list[ones]  = 0.
    wqb_list[zeros] = 1.

    ones  = wqb_list.eq(1.)
    zeros = wqb_list.eq(0.)

    for k in range(int(w_bitwidth/cellbit)):
        wqb    = wqb_list[k]
        is_one = ones[k]
        is_zrs = zeros[k]
        
        if k == 0:
            wqb[is_one]   = 0.
            wqb[is_zrs]   = 1.
            propagate_one = is_one
        else:
            wqb[is_one*propagate_one] = 0.
            wqb[is_zrs*propagate_one] = 1.
            propagate_one = is_one * propagate_one
        
        new_wqb_list[w_bitwidth - k] = wqb
    return new_wqb_list


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
    #print(prune_until, test_until)
    int_test_new = torch.zeros_like(int_test)

    for i, value in enumerate(int_test.tolist()):
        new_value = 0
        error = 1e7
        for n in range(2**(prune_until-test_until)):
            tmp_value = n * 2.**(w_bitwidth-prune_until)
            new_error = ((tmp_value - value)**2)
            #print('new error', new_error)
            if new_error < error:
                #print(tmp_value)
                error = new_error
                #print(error)
                new_value = tmp_value

        int_test_new[i] = new_value

    column_test_new = int_to_binary(int_test_new, w_bitwidth=w_bitwidth-test_until)
    group_binary[test_until:] = column_test_new
    group_int_new = signMagnitude_to_int(group_binary, w_bitwidth=w_bitwidth)

    if return_binary:
        return group_binary
    else:
        return group_int_new


def bitFlip_twosComplement(group_q, group_qb, w_bitwidth=8, zero_column_required=4, return_binary=False):
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
    hamming_distance = torch.sum(torch.abs(group_binary[tmp_idx] - group_binary[0]))
    if hamming_distance < 1.5:
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
        int_test = group_q
        int_test_new = torch.zeros_like(int_test)
        error = 1e7
        for value in range(-2**(w_bitwidth-2), 2**(w_bitwidth-2)):
            tmp_tensor = torch.Tensor([value for _ in range(group_q.shape[0])])
            new_error = torch.sum((tmp_tensor - int_test)**2)
            #print('new error', new_error)
            if new_error < error:
                error = new_error
                int_test_new = tmp_tensor
        group_int_new = int_test_new
    else:
        column_test = group_binary[prune_until:]
        int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-prune_until)
        int_test_new = torch.zeros_like(int_test)
        error = 1e7
        for value in range(2**(w_bitwidth-prune_until)):
            tmp_tensor = torch.Tensor([value for _ in range(group_q.shape[0])])
            new_error = torch.sum((tmp_tensor - int_test)**2)
            #print('new error', new_error)
            if new_error < error:
                #print(tmp_value)
                error = new_error
                #print(error)
                int_test_new = tmp_tensor
        column_test_new = int_to_binary(int_test_new, w_bitwidth=w_bitwidth-prune_until)
        group_binary[prune_until:] = column_test_new
        group_int_new = twosComplement_to_int(group_binary, w_bitwidth=w_bitwidth)

    if return_binary:
        return group_binary
    else:
        return group_int_new
    

def process_signMagnitude_conv(wq_int, w_bitwidth=8, group_size=16, pruned_column_num=4, device='cpu'):
    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_signMagnitude = wqb_signMagnitude.to('cpu')
    wq_int_new = torch.zeros_like(wq_int)
    K, C, W, H = wq_int.shape # output channel, input channel, kernel width, kernel height
    for k in range(K):  # output channel
        for w in range(W):  # kernel width
            for h in range(H):  # kernel height
                for c in range(C // group_size):  # input channel 
                    group_q = wq_int[k, c*group_size:(c+1)*group_size, w, h]
                    group_qb = wqb_signMagnitude[:, k, c*group_size:(c+1)*group_size, w, h]
                    group_q_new = bitFlip_signMagnitude(group_q, group_qb, w_bitwidth=w_bitwidth,
                                                        zero_column_required=pruned_column_num)
                    wq_int_new[k, c*group_size:(c+1)*group_size, w, h] = group_q_new
    return wq_int_new


def process_signMagnitude_fc(wq_int, w_bitwidth=8, group_size=16, pruned_column_num=4, device='cpu'):
    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_signMagnitude = wqb_signMagnitude.to('cpu')
    wq_int_new = torch.zeros_like(wq_int)
    K, C = wq_int.shape # output channel, input channel

    for k in range(K):  # output channel
        for c in range(C // group_size):  # input channel 
            group_q = wq_int[k, c*group_size:(c+1)*group_size]
            group_qb = wqb_signMagnitude[:, k, c*group_size:(c+1)*group_size]
            group_q_new = bitFlip_signMagnitude(group_q, group_qb, w_bitwidth=w_bitwidth,
                                                zero_column_required=pruned_column_num)
            wq_int_new[k, c*group_size:(c+1)*group_size] = group_q_new
    return wq_int_new


def process_twosComplement_conv(wq_int, w_bitwidth=8, group_size=16, pruned_column_num=4, device='cpu'):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_twosComplement = wqb_twosComplement.to('cpu')
    wq_int_new = torch.zeros_like(wq_int)
    K, C, W, H = wq_int.shape # output channel, input channel, kernel width, kernel height

    for k in range(K):  # output channel
        for w in range(W):  # kernel width
            for h in range(H):  # kernel height
                for c in range(C // group_size):  # input channel 
                    group_q = wq_int[k, c*group_size:(c+1)*group_size, w, h]
                    group_qb = wqb_twosComplement[:, k, c*group_size:(c+1)*group_size, w, h]
                    group_q_new = bitFlip_twosComplement(group_q, group_qb, w_bitwidth=w_bitwidth,
                                                         zero_column_required=pruned_column_num)
                    wq_int_new[k, c*group_size:(c+1)*group_size, w, h] = group_q_new
    return wq_int_new


def process_twosComplement_fc(wq_int, w_bitwidth=8, group_size=16, pruned_column_num=4, device='cpu'):
    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    wqb_twosComplement = wqb_twosComplement.to('cpu')
    wq_int_new = torch.zeros_like(wq_int)
    K, C = wq_int.shape # output channel, input channel, kernel width, kernel height

    for k in range(K):  # output channel
        for c in range(C // group_size):  # input channel 
            group_q = wq_int[k, c*group_size:(c+1)*group_size]
            group_qb = wqb_twosComplement[:, k, c*group_size:(c+1)*group_size]            
            group_q_new = bitFlip_twosComplement(group_q, group_qb, w_bitwidth=w_bitwidth,
                                                 zero_column_required=pruned_column_num)
            wq_int_new[k, c*group_size:(c+1)*group_size] = group_q_new
    return wq_int_new
