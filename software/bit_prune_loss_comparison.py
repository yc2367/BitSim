import torch, torchvision
import torch.nn as nn
from util.bitflip_layer import *
import time
import csv, json
import argparse
from statistics import mean

from torchvision.models.quantization import (ResNet18_QuantizedWeights, 
                                             MobileNet_V2_QuantizedWeights,
                                             ResNet50_QuantizedWeights)

class Profiler:
    def __init__(self, model_name:str) -> None:
        self.layer_name_list = self._extract_layer_name(model_name)
        self.model_q = self._get_quantized_model(model_name)

    def _extract_layer_name(self, model_name: str):
        layer_name_list = []
        base_path = '/home/yc2367/BitVert_DNN'
        model_config_path = f'{base_path}/Baseline_Int8/{model_name}'
        layer_dim_path = f'{model_config_path}/tensors/matmul.json'

        with open(layer_dim_path) as f:
            layer_dim_list = json.load(f)
        
        for name, _ in layer_dim_list.items():
            layer_name = name.rstrip('.ops')
            '''
            if ('fc' in layer_name) or ('classifier' in layer_name) or \
                ('proj' in layer_name) or ('qkv' in layer_name) or ('conv' in layer_name):
                layer_name_list.append(layer_name)
            '''
            if ('fc' in layer_name) or ('conv' in layer_name):
                layer_name_list.append(layer_name)
        return layer_name_list

    def _get_quantized_model(self, model_name):
        model_q = []
        base_path = '/home/yc2367/BitVert_DNN'
        model_config_path = f'{base_path}/Baseline_Int8/{model_name}'
        tensor_path = f'{model_config_path}/tensors'

        for name in self.layer_name_list:
            w_tensor_file = f'{tensor_path}/{name}.ops_x2.pt'
            w_tensor = torch.load(w_tensor_file)
            if len(w_tensor.shape) == 2: # transpose fully-connected layer
                w_tensor = w_tensor.permute(1, 0)
            model_q.append(w_tensor)
        return model_q


parser = argparse.ArgumentParser()
parser.add_argument('--model')
args = parser.parse_args()

model_name = args.model
profiler = Profiler(model_name)
weight_list = profiler.model_q
name_list = profiler.layer_name_list

GROUP_SIZE = 32
w_bitwidth = 8
num_col_pruned = 4

loss = 1
if loss == 0:
    metric = 'MSE'
else: 
    metric = 'KL_DIV'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    bitwave_loss = ['BitWave']
    bitvert_loss_ra = ['Round Avg']
    bitvert_loss_zp = ['ZP Shifting']
    
    for N in range(num_col_pruned, num_col_pruned+1):
        pruned_column_num = N
        file = open(f'{model_name}_loss_report_g{GROUP_SIZE}_c{pruned_column_num}.txt', 'w')

        start = time.time()
        for i in range(len(weight_list) // 2 + 1, len(weight_list)):
            weight_test = weight_list[i]
            weight_shape = weight_list[i].shape
            print(f'Layer {name_list[i]}')
            print(f'Layer Shape: {weight_shape}')
            file.writelines(f'Layer {name_list[i]} \n')
            file.writelines(f'Layer Shape: {weight_shape} \n')
            
            #print(weight_test.unique())
            for func in [0, 1, 2,]:
                if func == 0:
                    format = 'Round to Nearest'
                    if len(weight_test.shape) == 4:
                        weight_test_new = bitflip_signMagnitude_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    num_pruned_column=pruned_column_num, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = bitflip_signMagnitude_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                num_pruned_column=pruned_column_num, device=device)
                    #print(weight_test_new.unique())
                elif func == 1:
                    format = 'Column Averaging'
                    if len(weight_test.shape) == 4:
                        #weight_test = weight_test[2:3,0:16, 0:1,0:1]
                        weight_test_new = colAvg_twosComplement_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                      num_pruned_column=pruned_column_num, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = colAvg_twosComplement_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    num_pruned_column=pruned_column_num, device=device)
                elif func == 2:
                    format = 'Zero Point'
                    if len(weight_test.shape) == 4:
                        weight_test_new = bitflip_zeroPoint_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                 num_pruned_column=pruned_column_num, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = bitflip_zeroPoint_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                               num_pruned_column=pruned_column_num, device=device)
                    #print(weight_test_new.unique())
                else:
                    format = 'Optimal'
                    if len(weight_test.shape) == 4:
                        weight_test_new = bitVert_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                        num_pruned_column=pruned_column_num, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = bitVert_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                    num_pruned_column=pruned_column_num, device=device)

                weight_original = weight_test.to(dtype=torch.float, device=device)
                weight_new = weight_test_new.to(device)

                if metric == 'MSE':
                    criterion = nn.MSELoss()
                    loss = criterion(weight_original, weight_new)
                else:
                    criterion = nn.KLDivLoss();
                    weight_original = F.log_softmax(weight_original.reshape(-1), dim=0)
                    weight_new = F.softmax(weight_new.reshape(-1), dim=0)
                    loss = criterion(weight_original, weight_new)
                
                loss = loss.item()
                if func == 0:
                    bitwave_loss.append(loss)
                elif func == 1:
                    bitvert_loss_ra.append(loss)
                elif func == 2:
                    bitvert_loss_zp.append(loss)
                #print(f'{format}: MSE loss between new weight and original weight is {loss}')
                print(f'{format.ljust(20)} {metric.ljust(8)}: {loss}')
                file.writelines(f'{format.ljust(20)} {metric.ljust(8)}: {loss} \n')
            print()
            file.writelines('\n')
        file.close()

        end = time.time()
        print(f'It took {end - start} seconds!')

        loss_comparison = [bitwave_loss, bitvert_loss_ra, bitvert_loss_zp]
        with open('bitwave_bitvert_loss_comparison.txt', 'w') as f:
            write = csv.writer(f)
            write.writerows(loss_comparison)



if __name__ == "__main__":
    main()
