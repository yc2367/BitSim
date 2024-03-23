import torch, torchvision
import torch.nn as nn
import numpy as np
import math
from util.bitflip_layer import *
import time

from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
model = torchvision.models.quantization.mobilenet_v2(weights = MobileNet_V2_QuantizedWeights, 
                                                     quantize=True)

model = model.cpu()

weight_list = []
name_list   = []

for n, m in model.named_modules():
    if hasattr(m, "weight"):
        w = m.weight()
        wint = torch.int_repr(w)
        weight_list.append(wint)
        name_list.append(n)

GROUP_SIZE = 16
w_bitwidth = 8
num_col_pruned = 4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    for N in range(num_col_pruned, num_col_pruned+1):
        pruned_column_num = N
        file = open(f'MobileNet_loss_report_g{GROUP_SIZE}_c{pruned_column_num}.txt', 'w')

        start = time.time()
        for i in range(1, len(weight_list)):
            weight_test = weight_list[i]
            print(f'Layer {name_list[i]}')
            print(f'Layer Shaoe {weight_test.shape}')
            print(f'Layer Params {torch.prod(torch.Tensor(list(weight_test.shape)))}')
            file.writelines(f'Layer {name_list[i]} \n')
            file.writelines(f'Layer Shaoe {weight_test.shape} \n')
            file.writelines(f'Layer Params {torch.prod(torch.Tensor(list(weight_test.shape)))} \n')
            # print(weight_test.unique())
            for func in [0, 1, 2,]:
                if func == 0:
                    format = 'Sign Magnitude'
                    if len(weight_test.shape) == 4:
                        weight_test_new = bitflip_signMagnitude_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    num_pruned_column=pruned_column_num, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = bitflip_signMagnitude_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                num_pruned_column=pruned_column_num, device=device)
                    #print(weight_test_new.unique())
                elif func == 1:
                    format = '2s Complement'
                    if len(weight_test.shape) == 4:
                        #weight_test = weight_test[2:3,0:16, 0:1,0:1]
                        weight_test_new = colAvg_twosComplement_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                      num_pruned_column=pruned_column_num, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = colAvg_twosComplement_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    num_pruned_column=pruned_column_num, device=device)
                elif func == 2:
                    format = 'ZP SignMagnitude'
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

                loss = criterion(weight_original, weight_new)
                #print(f'{format}: MSE loss between new weight and original weight is {loss}')
                print(f'{format.ljust(20)} MSE: {loss}')
                file.writelines(f'{format.ljust(20)} MSE: {loss} \n')
            print()
            file.writelines('\n')
        file.close()

        end = time.time()
        print(f'It took {end - start} seconds!')

                    
if __name__ == "__main__":
    main()
