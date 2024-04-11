import torch
import numpy as np
from util.count_sparsity import *
import argparse, json

parser = argparse.ArgumentParser()
parser.add_argument('--model')
args = parser.parse_args()

model_name = args.model
    
def extract_weight_tensor(model_name: str):
    base_path = '/home/yc2367/BitVert_DNN'
    model_config_path = f'{base_path}/Baseline_Int8/{model_name}'
    tensor_path = f'{model_config_path}/tensors'
    layer_dim_path = f'{model_config_path}/tensors/matmul.json'

    with open(layer_dim_path) as f:
        layer_dim_list = json.load(f)
    
    weight_tensor_dict = {}
    for layer_name, _ in layer_dim_list.items():
        if 'conv' in layer_name:
            weight_tensor_dict[layer_name] = torch.load(f'{tensor_path}/{layer_name}_x2.pt')
        elif ('fc' in layer_name) or ('proj' in layer_name) or \
            ('qkv' in layer_name) or ('classifier' in layer_name) :
            weight_tensor_dict[layer_name] = torch.load(f'{tensor_path}/{layer_name}_x2.pt')

    return weight_tensor_dict


def main():
    weight_tensor_dict = extract_weight_tensor(model_name)
    GROUP_SIZE = 8
    w_bitwidth = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file = open(f'{model_name}_sparsity_group_{GROUP_SIZE}.txt', 'w')

    sparse_value_count = 0
    sparse_bit_count_sm = 0
    sparse_bit_count_2s = 0
    sparse_bit_count_skip_0_col_sm = 0
    sparse_bit_count_skip_01_2s = 0
    sparse_bit_count_skip_01_msb = 0

    total_bit_count_model = 0
    total_value_count_model = 0
    total_value_count_until_now  = 0

    count_zero_column = CountZeroColumn()

    for name, weight_test in weight_tensor_dict.items():
        layer_total_value  = 0
        if len(weight_test.shape) == 4:
            _, layer_total_value = count_zero_value_conv(weight_test)
        elif len(weight_test.shape) == 2:
            _, layer_total_value = count_zero_value_fc(weight_test)
        total_value_count_model += layer_total_value

    for name, weight_test in weight_tensor_dict.items():
        print(weight_test.shape)
        print(f'Layer {name}')
        file.writelines(f'Layer {name} \n')
        file.writelines(f'Shape {weight_test.shape} \n')
        for func in [0, 1, 2, 3, 4, 5]:
            if func == 0:
                layer_sparse_value = 0
                layer_total_value  = 0
                format = 'Skip Zero Value'
                if len(weight_test.shape) == 4:
                    layer_sparse_value, layer_total_value = count_zero_value_conv(weight_test)
                elif len(weight_test.shape) == 2:
                    layer_sparse_value, layer_total_value = count_zero_value_fc(weight_test)
                sparse_value_count += layer_sparse_value
                total_value_count_until_now += layer_total_value

                line = f'{format.ljust(25)} Sparse value count: {layer_sparse_value}'
                print(line)
                file.writelines(f'{line} \n')

                line = f'Layer value count: {layer_total_value}'
                print(line)
                file.writelines(f'{line} \n')

                line = f'Layer value percent: {layer_total_value / total_value_count_model * 100}%'
                print(line)
                file.writelines(f'{line} \n')

                line = f'Value percent until now: {total_value_count_until_now / total_value_count_model * 100}%'
                print(line, '\n')
                file.writelines(f'{line} \n')
                file.writelines('\n')
            else:
                layer_sparse_bit = 0
                if func == 1:
                    format = 'Skip 0 bit, Sign Mag'
                    if len(weight_test.shape) == 4:
                        layer_sparse_bit, layer_total_bit = count_zero_bit_sm_conv(weight_test, w_bitwidth=w_bitwidth, device=device)
                    elif len(weight_test.shape) == 2:
                        layer_sparse_bit, layer_total_bit = count_zero_bit_sm_fc(weight_test, w_bitwidth=w_bitwidth, device=device)
                    sparse_bit_count_sm += layer_sparse_bit
                    total_bit_count_model += layer_total_bit
                elif func == 2:
                    format = 'Skip bbs col, 2s Comp'
                    if len(weight_test.shape) == 4 and weight_test.shape[1] != 1:
                        layer_sparse_bit = count_bbs_column_conv(weight_test, w_bitwidth=w_bitwidth, 
                                                                group_size=GROUP_SIZE, device=device)
                    elif len(weight_test.shape) == 2 and weight_test.shape[1] != 1:
                        layer_sparse_bit = count_bbs_column_fc(weight_test, w_bitwidth=w_bitwidth, 
                                                                group_size=GROUP_SIZE, device=device)
                    sparse_bit_count_skip_0_col_sm += layer_sparse_bit
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
    file.writelines(f'{line} \n')
    line = f'Model total value count: {total_value_count_model}\n'
    print(line)
    file.writelines(f'{line} \n\n')

    format = 'Skip 0 bit, Sign Mag'
    line = f'{format.ljust(25)} Total sparse bit count: {sparse_bit_count_sm}'
    print(line)
    file.writelines(f'{line} \n')

    format = 'Skip bbs col, 2s Comp'
    line = f'{format.ljust(25)} Total sparse bit count: {sparse_bit_count_skip_0_col_sm}'
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
