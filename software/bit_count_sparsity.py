import torch
import torch.nn as nn
import torchvision
import numpy as np
from util.count_sparsity import *
import argparse

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
        w = m.weight()
        wint = torch.int_repr(w)
        weight_list.append(wint)
        name_list.append(n)

GROUP_SIZE = 4
w_bitwidth = 8

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file = open(f'{model_name}_sparsity_group_{GROUP_SIZE}.txt', 'w')

    sparse_value_count = 0
    sparse_bit_count_sm = 0
    sparse_bit_count_2s = 0
    sparse_bit_count_skip_01_sm = 0
    sparse_bit_count_skip_01_2s = 0
    sparse_bit_count_skip_01_msb = 0

    total_bit_count_model = 0
    total_value_count_model = 0

    for i in range(len(weight_list)):
        weight_test = weight_list[i]
        print(weight_test.shape)
        print(f'Layer {name_list[i]}')
        file.writelines(f'Layer {name_list[i]} \n')
        for func in [0, 1, 2, 3, 4, 5]:
            if func == 0:
                layer_sparse_value = 0
                layer_total_value  = 0
                format = 'Skip Zero Value'
                if len(weight_test.shape) == 4 and weight_test.shape[1] != 1:
                    layer_sparse_value, layer_total_value = count_zero_value_conv(weight_test)
                elif len(weight_test.shape) == 2:
                    layer_sparse_value, layer_total_value = count_zero_value_fc(weight_test)
                sparse_value_count += layer_sparse_value
                total_value_count_model += layer_total_value

                line = f'{format.ljust(25)} Sparse value count: {layer_sparse_value}'
                print(line)
                file.writelines(f'{line} \n')
                line = f'Layer value count: {layer_total_value}'
                print(line, '\n')
                file.writelines(f'{line} \n')
                file.writelines('\n\n')
            else:
                layer_sparse_bit = 0
                layer_total_bit  = 0
                if func == 1:
                    format = 'Skip 0 bit, Sign Mag'
                    if len(weight_test.shape) == 4 and weight_test.shape[1] != 1:
                        layer_sparse_bit, layer_total_bit = count_zero_bit_sm_conv(weight_test, w_bitwidth=w_bitwidth, device=device)
                    elif len(weight_test.shape) == 2:
                        layer_sparse_bit, layer_total_bit = count_zero_bit_sm_fc(weight_test, w_bitwidth=w_bitwidth, device=device)
                    sparse_bit_count_sm += layer_sparse_bit
                    total_bit_count_model += layer_total_bit
                elif func == 2:
                    format = 'Skip 1 or 0 bit, SM'
                    if len(weight_test.shape) == 4 and weight_test.shape[1] != 1:
                        layer_sparse_bit, _ = count_less_bit_sm_conv(weight_test, w_bitwidth=w_bitwidth, 
                                                                group_size=GROUP_SIZE, device=device)
                    elif len(weight_test.shape) == 2:
                        layer_sparse_bit, _ = count_less_bit_sm_fc(weight_test, w_bitwidth=w_bitwidth, 
                                                                group_size=GROUP_SIZE, device=device)
                    sparse_bit_count_skip_01_sm += layer_sparse_bit
                elif func == 3:
                    format = 'Skip 0 bit, 2s Comp'
                    if len(weight_test.shape) == 4 and weight_test.shape[1] != 1:
                        layer_sparse_bit, _ = count_zero_bit_2s_conv(weight_test, w_bitwidth=w_bitwidth, device=device)
                    elif len(weight_test.shape) == 2:
                        layer_sparse_bit, _ = count_zero_bit_2s_fc(weight_test, w_bitwidth=w_bitwidth, device=device)
                    sparse_bit_count_2s += layer_sparse_bit
                elif func == 4:
                    format = 'Skip 1 or 0 bit, 2s'
                    if len(weight_test.shape) == 4 and weight_test.shape[1] != 1:
                        layer_sparse_bit, _ = count_less_bit_2s_conv(weight_test, w_bitwidth=w_bitwidth, 
                                                                group_size=GROUP_SIZE, device=device)
                    elif len(weight_test.shape) == 2:
                        layer_sparse_bit, _ = count_less_bit_2s_fc(weight_test, w_bitwidth=w_bitwidth, 
                                                                group_size=GROUP_SIZE, device=device)
                    sparse_bit_count_skip_01_2s += layer_sparse_bit
                elif func == 5:
                    format = 'Proposed msb clipping'
                    if len(weight_test.shape) == 4 and weight_test.shape[1] != 1:
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
        print(line)
        file.writelines(f'{line} \n')

        print('\n')
        file.writelines(f'\n\n')

    
    format = 'Skip Zero Value'
    line = f'{format.ljust(25)} Total sparse value count: {sparse_value_count}'
    print(line)
    file.writelines(f'{line} \n\n')
    line = f'Model total value count: {total_value_count_model}\n'
    print(line)
    file.writelines(f'{line} \n\n')

    format = 'Skip 0 bit, Sign Mag'
    line = f'{format.ljust(25)} Total sparse bit count: {sparse_bit_count_sm}'
    print(line)
    file.writelines(f'{line} \n')

    format = 'Skip 1 or 0 bit, SM'
    line = f'{format.ljust(25)} Total sparse bit count: {sparse_bit_count_skip_01_sm}'
    print(line)
    file.writelines(f'{line} \n')

    format = 'Skip 0 bit, 2s Comp'
    line = f'{format.ljust(25)} Total sparse bit count: {sparse_bit_count_2s}'
    print(line)
    file.writelines(f'{line} \n')

    format = 'Skip 1 or 0 bit, 2s'
    line = f'{format.ljust(25)} Total sparse bit count: {sparse_bit_count_skip_01_2s}'
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
