import torch
import torch.nn as nn
import torchvision
import numpy as np
from util import *

from torchvision.models.quantization import ResNet50_QuantizedWeights
model = torchvision.models.quantization.resnet50(weights = ResNet50_QuantizedWeights, quantize=True)

model = model.cpu()

weight_list = []
name_list   = []

for n, m in model.named_modules():
    if hasattr(m, "weight"):
        w = m.weight()
        wint = torch.int_repr(w)
        weight_list.append(wint)
        name_list.append(n)

GROUP_SIZE = 8
w_bitwidth = 8

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file = open(f'resnet50_bit_count_msb_clipping_group_{GROUP_SIZE}.txt', 'w')
    sparse_bit_count_baseline = 0
    sparse_bit_count_skip_01 = 0
    sparse_bit_count_skip_01_msb = 0
    total_bit_count_model = 0

    for i in range(len(weight_list)):
        weight_test = weight_list[i]
        print(weight_test.shape)
        print(f'Layer {name_list[i]}')
        file.writelines(f'Layer {name_list[i]} \n')
        for func in [0, 1, 2]:
            if func == 0:
                format = 'Baseline skip 0'
                if len(weight_test.shape) == 4:
                    layer_sparse_bit, layer_total_bit = count_zero_bit_conv(weight_test, w_bitwidth=w_bitwidth, 
                                                                        group_size=GROUP_SIZE, device=device)
                elif len(weight_test.shape) == 2:
                    layer_sparse_bit, layer_total_bit = count_zero_bit_fc(weight_test, w_bitwidth=w_bitwidth, 
                                                                        group_size=GROUP_SIZE, device=device)
                sparse_bit_count_baseline += layer_sparse_bit
                total_bit_count_model += layer_total_bit
            elif func == 1:
                format = 'Proposed skip 1 or 0'
                if len(weight_test.shape) == 4:
                    layer_sparse_bit, _ = count_less_bit_conv(weight_test, w_bitwidth=w_bitwidth, 
                                                            group_size=GROUP_SIZE, device=device)
                elif len(weight_test.shape) == 2:
                    layer_sparse_bit, _ = count_less_bit_fc(weight_test, w_bitwidth=w_bitwidth, 
                                                            group_size=GROUP_SIZE, device=device)
                sparse_bit_count_skip_01 += layer_sparse_bit
            elif func == 2:
                format = 'Proposed msb clipping'
                if len(weight_test.shape) == 4:
                    layer_sparse_bit, _ = count_less_bit_clip_msb_conv(weight_test, w_bitwidth=w_bitwidth, 
                                                                    group_size=GROUP_SIZE, device=device)
                elif len(weight_test.shape) == 2:
                    layer_sparse_bit, _ = count_less_bit_clip_msb_fc(weight_test, w_bitwidth=w_bitwidth, 
                                                                    group_size=GROUP_SIZE, device=device)
                sparse_bit_count_skip_01_msb += layer_sparse_bit
            
            line = f'{format.ljust(25)} Sparse bit count: {layer_sparse_bit}'
            print(line)
            file.writelines(f'{line} \n')
        line = f'Layer bit count: {layer_total_bit}'
        print(line, '\n')
        file.writelines(f'{line} \n')
        file.writelines('\n')
    
    format = 'Baseline skip 0'
    line = f'{format.ljust(25)} Total sparse bit count: {sparse_bit_count_baseline}'
    print(line)
    file.writelines(f'{line} \n')
    format = 'Proposed skip 1 or 0'
    line = f'{format.ljust(25)} Total sparse bit count: {sparse_bit_count_skip_01}'
    print(line)
    file.writelines(f'{line} \n')
    format = 'Proposed msb clipping'
    line = f'{format.ljust(25)} Total sparse bit count: {sparse_bit_count_skip_01_msb}'
    print(line)
    file.writelines(f'{line} \n')
    line = f'Model total bit count: {total_bit_count_model}'
    print(line)
    file.writelines(f'{line} \n')
    file.writelines('\n')
    file.close()

                    
if __name__ == "__main__":
    main()