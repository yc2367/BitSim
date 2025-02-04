import torch
import torch.nn as nn
import torchvision
import numpy as np
from util.count_sparsity import *
import argparse, math, time

from torchvision.models.quantization import (ResNet18_QuantizedWeights, 
                                             MobileNet_V2_QuantizedWeights,
                                             ResNet50_QuantizedWeights)

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices = ['resnet18', 'resnet50', 'mobilenet_v2'])
args = parser.parse_args()

model_name = args.model

if model_name == 'resnet18':
    weights = ResNet18_QuantizedWeights
    model = torchvision.models.quantization.resnet18(weights = weights, quantize=True)
elif model_name == 'resnet50':
    weights = ResNet50_QuantizedWeights
    model = torchvision.models.quantization.resnet50(weights = weights, quantize=True)
elif model_name == 'mobilenet_v2':
    weights = MobileNet_V2_QuantizedWeights
    model = torchvision.models.quantization.mobilenet_v2(weights = weights, quantize=True)
else:
    raise ValueError('ERROR! The provided model is not one of supported models.')

model = model.cpu()

weight_list = []
name_list   = []
for n, m in model.named_modules():
    if hasattr(m, "weight"):
        print(m)
        w = m.weight()
        wint = torch.int_repr(w)
        weight_list.append(wint)
        name_list.append(n)

w_bitwidth = 8

def count_cycle_bbs_clip_msb_conv(wq_int, w_bitwidth=8, group_size=32, pe_row=32, device='cpu'):
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
        sub_group_size = C
    else:
        sub_group_size = 8
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([2, 3, 0, 1]).unsqueeze(-1)
    wq_int = wq_int.reshape(H, W, K, C//group_size, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    workload = wqb_twosComplement.reshape(w_bitwidth, H, W, K, C//group_size, group_size // sub_group_size, sub_group_size)

    # tensor to store msb index of all groups
    msb_idx = torch.zeros([H, W, K, C//group_size, group_size // sub_group_size], device=device) 
    eq_msb_column = torch.ones([H, W, K, C//group_size, group_size // sub_group_size], device=device)
    for i in range(1, int(w_bitwidth)):
        eq_column = torch.all(torch.eq(workload[0], workload[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        msb_idx[eq_msb_column] += 1
    
    for i in range(1, int(w_bitwidth)):
        set_column_to_zero = msb_idx.ge(i)
        workload[i][set_column_to_zero] = 0

    bit_one_count = torch.sum(workload, dim=-1)
    skip_zero = bit_one_count.gt(sub_group_size/2)
    bit_one_count[skip_zero] = sub_group_size - bit_one_count[skip_zero]

    cycle_total = 0
    for h in range(H):
        for w in range(W):
            for i in range(math.ceil(K/pe_row)):
                for n in range(C//group_size):
                    l_idx = i * pe_row
                    u_idx = (i+1) * pe_row

                    load = bit_one_count[:, h, w, l_idx:u_idx, n, :]
                    cycle_load = torch.ceil(load / 3)
                    
                    cycle_load = torch.sum(cycle_load, dim=0)
                    cycle_load = torch.max(cycle_load).item()
                    cycle_total += cycle_load
    return cycle_total

def count_cycle_clip_msb_conv(wq_int, w_bitwidth=8, group_size=32, pe_row=16, device='cpu'):
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
        sub_group_size = C
    else:
        sub_group_size = 32
    wq_int = wq_int.permute([2, 3, 0, 1]).unsqueeze(-1)
    wq_int = wq_int.reshape(H, W, K, C//group_size, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    workload = wqb_twosComplement.reshape(w_bitwidth, H, W, K, C//group_size, group_size // sub_group_size, sub_group_size)

    # tensor to store msb index of all groups
    msb_idx = torch.zeros([H, W, K, C//group_size, group_size // sub_group_size], device=device) 
    eq_msb_column = torch.ones([H, W, K, C//group_size, group_size // sub_group_size], device=device)
    for i in range(1, int(w_bitwidth)):
        eq_column = torch.all(torch.eq(workload[0], workload[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        msb_idx[eq_msb_column] += 1
    
    for i in range(1, int(w_bitwidth)):
        set_column_to_zero = msb_idx.ge(i)
        workload[i][set_column_to_zero] = 0

    bit_one_count = torch.sum(workload, dim=-1)
    skip_zero = bit_one_count.gt(sub_group_size/2)
    bit_one_count[skip_zero] = sub_group_size - bit_one_count[skip_zero]

    cycle_total = 0
    for h in range(H):
        for w in range(W):
            for i in range(math.ceil(K/pe_row)):
                for n in range(C//group_size):
                    l_idx = i * pe_row
                    u_idx = (i+1) * pe_row

                    load = bit_one_count[:, h, w, l_idx:u_idx, n, :]
                    cycle_load = torch.ceil(load / 16)
                    
                    cycle_load = torch.sum(cycle_load, dim=0)
                    cycle_load = torch.max(cycle_load).item()
                    cycle_total += cycle_load
    return cycle_total

def count_cycle_conv(wq_int, w_bitwidth=8, group_size=32, pe_row=16, device='cpu'):
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C

    NUM_GROUP = K*W*H*C//group_size

    cycle = NUM_GROUP / pe_row * w_bitwidth
    return cycle

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_cycle_normal = 0
    total_cycle_enhanced = 0
    
    for i in range(1, len(weight_list)-1):
        weight_test = weight_list[i]
        print(weight_test.shape)
        cycle_normal = count_cycle_conv(weight_test, pe_row=16, device=device)
        cycle_enhanced = count_cycle_clip_msb_conv(weight_test, pe_row=16, device=device)
        total_cycle_normal += cycle_normal
        total_cycle_enhanced += cycle_enhanced
        print(name_list[i])
        print(f'normal:   {cycle_normal}')
        print(f'enhanced: {cycle_enhanced}')
        print()
    
    print('\n')
    print(f'normal:   {total_cycle_normal}')
    print(f'enhanced: {total_cycle_enhanced}')
        

        


                    
if __name__ == "__main__":
    main()
