import torch
import torch.nn as nn


def int_to_binary(weight_q, w_bitwidth: int=8, device='cpu'):
    weight_q = weight_q.clone()
    
    weight_q_shape = list(weight_q.size())
    remainder_list = torch.zeros([w_bitwidth] + weight_q_shape, device=device)
    for k in reversed(range(w_bitwidth)):
        remainder = torch.fmod(weight_q, 2)
        remainder_list[k] = remainder
        weight_q = torch.round((weight_q-remainder)/2)
    return remainder_list


def binary_to_int(wqb_list, w_bitwidth=8, device='cpu'):
    bin_list_shape = wqb_list.size()
    wq_list_shape = list(bin_list_shape[1:])
    wq_list = torch.zeros(wq_list_shape, device=device)
    for k in reversed(range(int(w_bitwidth))):
        wq_list += (wqb_list[k] * 2.**(w_bitwidth-1-k))
    return wq_list
    

def int_to_signMagnitude(weight_q, w_bitwidth: int=8, device='cpu'):
    weight_q = weight_q.to(device, copy=True)
    is_min = weight_q.eq(-2**(w_bitwidth-1))
    weight_q[is_min] = -2**(w_bitwidth-1) + 1
    is_neg = weight_q.lt(0.)
    weight_q[is_neg] = -weight_q[is_neg]
    
    weight_q_shape = list(weight_q.size())
    remainder_list = torch.zeros([w_bitwidth] + weight_q_shape, device=device)

    for k in reversed(range(w_bitwidth)):
        remainder = torch.fmod(weight_q, 2)
        remainder_list[k] = remainder
        weight_q = torch.round((weight_q-remainder)/2)

    remainder_list[0, is_neg] = 1.
    return remainder_list


def signMagnitude_to_int(wqb_list, w_bitwidth: int=8, device='cpu'):
    bin_list_shape = wqb_list.size()
    wq_list_shape = list(bin_list_shape[1:])
    wq_list = torch.zeros(wq_list_shape, device=device)

    for k in reversed(range(int(w_bitwidth))):
        if k != 0:
          wq_list += (wqb_list[k] * 2.**(w_bitwidth-1-k))
        else:
          ones  = wqb_list[k].eq(1.)
          wq_list[ones] = -wq_list[ones]
    return wq_list


def int_to_twosComplement(weight_q, w_bitwidth: int=8, device='cpu'):
    weight_q = weight_q.to(device, copy=True)
    is_neg = weight_q.lt(0.)
    weight_q[is_neg] = 2**(w_bitwidth-1) + weight_q[is_neg]

    weight_q_shape = list(weight_q.size())
    remainder_list = torch.zeros([w_bitwidth] + weight_q_shape, device=device)
    for k in reversed(range(w_bitwidth)):
        remainder = torch.fmod(weight_q, 2)
        remainder_list[k] = remainder
        weight_q = torch.round((weight_q-remainder)/2)
    
    remainder_list[0, is_neg] = 1.
    return remainder_list


def twosComplement_to_int(wqb_list, w_bitwidth: int=8, device='cpu'):
    bin_list_shape = wqb_list.size()
    wq_list_shape = list(bin_list_shape[1:])
    wq_list = torch.zeros(wq_list_shape, device=device)

    for k in reversed(range(int(w_bitwidth))):
        if k != 0:
          wq_list += (wqb_list[k] * 2.**(w_bitwidth-1-k))
        else:
          wq_list -= (wqb_list[k] * 2.**(w_bitwidth-1-k))

    return wq_list

def less_bit_twosComplement_conv(wq_int, w_bitwidth=8, group_size=32, device='cpu'):
    K, C, W, H = wq_int.shape # output channel, input channel, kernel width, kernel height        
    if C < group_size:
        group_size = C
        return int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    bit_one_count = torch.sum(wqb_twosComplement, dim=-1)
    skip_one = bit_one_count.gt(group_size/2)
    wqb_twosComplement[skip_one, :] = 1 - wqb_twosComplement[skip_one, :]

    return wqb_twosComplement.view(8, K, W, H, C).permute(0, 1, 4, 2, 3)


def less_bit_twosComplement_fc(wq_int, w_bitwidth=8, group_size=32, device='cpu'):
    K, C = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    bit_one_count = torch.sum(wqb_twosComplement, dim=-1)
    skip_one = bit_one_count.gt(group_size/2)
    print(wqb_twosComplement[skip_one, :][0])
    wqb_twosComplement[skip_one, :] = 1 - wqb_twosComplement[skip_one, :]
    print(wqb_twosComplement[skip_one, :][0])

    return wqb_twosComplement.view(8, K, C)