import torch

def int_to_signMagnitude(weight_q, w_bitwidth=8, cellBit=1):
    weight_q = weight_q.clone()

    cellRange = 2**cellBit
    is_min = weight_q.eq(-2**(w_bitwidth-1))
    weight_q[is_min] = -2**(w_bitwidth-1) + 1
    is_neg = weight_q.lt(0.)
    weight_q[is_neg] = -weight_q[is_neg]
    
    weight_q_shape = torch.Tensor(list(weight_q.size()))
    bin_list_shape = torch.cat((torch.Tensor([w_bitwidth/cellBit]), weight_q_shape)).to(int).tolist()
    remainder_list = torch.zeros(bin_list_shape).type_as(weight_q)

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
          wq_list += wqb_list[k] * 2**(w_bitwidth-1-k)
        else:
          ones  = wqb_list[k].eq(1.)
          wq_list[ones] = -wq_list[ones]
    return wq_list


def int_to_twosComplement(weight_q, w_bitwidth=8, cellBit=1):
    weight_q = weight_q.clone()

    cellRange = 2**cellBit
    is_neg = weight_q.lt(0.)
    weight_q[is_neg] = 2**(w_bitwidth-1) + weight_q[is_neg]
    
    weight_q_shape = torch.Tensor(list(weight_q.size()))
    bin_list_shape = torch.cat((torch.Tensor([w_bitwidth/cellBit]), weight_q_shape)).to(int).tolist()
    remainder_list = torch.zeros(bin_list_shape).type_as(weight_q)

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
          wq_list += wqb_list[k] * 2**(w_bitwidth-1-k)
        else:
          wq_list -= wqb_list[k] * 2**(w_bitwidth-1-k)
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
        wq_list += wqb_list[k] * 2**(w_bitwidth-1-k)
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


def bitFlip_signMagnitude(group_q, w_bitwidth=8, zero_column_required=4):
    '''
    Apply bit-flip to a group of quantized weights in sign-magnitude format
    '''
    group_binary = int_to_signMagnitude(group_q, w_bitwidth=w_bitwidth)
    # check existing zero columns
    zero_column_idx = [1e5]
    for i in range(1, int(w_bitwidth)):
        if torch.sum(group_binary[i]) == 0:
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
        error = 1e5
        for n in range(2**(prune_until-test_until)):
            tmp_value = n * 2**(w_bitwidth-prune_until)
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

    return group_binary


def process_signMagnitude_conv(wq_list, w_bitwidth=8, group_size=16, pruned_column_num=4):
    wq_list_new = torch.zeros_like(wq_list)
    K, C, W, H = wq_list.shape # output channel, input channel, kernel width, kernel height

    for k in range(K):  # output channel
        for w in range(W):  # kernel width
            for h in range(H):  # kernel height
                for c in range(C // group_size):  # input channel 
                    group_q = wq_list[k, c*group_size:(c+1)*group_size, w, h]
                    group_q_new = bitFlip_signMagnitude(group_q, w_bitwidth=w_bitwidth, zero_column_required=pruned_column_num)
                    wq_list_new[k, c*group_size:(c+1)*group_size, w, h] = group_q_new
    return wq_list_new