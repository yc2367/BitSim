import torch, torchvision
import torch.nn as nn
import math
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

GROUP_SIZE = 16
w_bitwidth = 8

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hamming_distance = 1.5
    for N in range(6, 7):
        pruned_column_num = N
        file = open(f'resnet50_loss_report_g{GROUP_SIZE}_h{math.floor(hamming_distance)}_c{pruned_column_num}.txt', 'w')

        for i in range(48, 54):
            weight_test = weight_list[i]
            print(f'Layer {name_list[i]}')
            file.writelines(f'Layer {name_list[i]} \n')
            for func in [0, 1]:
                if func == 0:
                    format = 'Sign Magnitude'
                    if len(weight_test.shape) == 4:
                        weight_test_new = process_signMagnitude_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    pruned_column_num=pruned_column_num, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = process_signMagnitude_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                pruned_column_num=pruned_column_num, device=device)
                else:
                    format = '2s Complement'
                    if len(weight_test.shape) == 4:
                        weight_test_new = process_twosComplement_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    pruned_column_num=pruned_column_num, device=device, 
                                                                    h_distance_target=hamming_distance)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = process_twosComplement_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    pruned_column_num=pruned_column_num, device=device,
                                                                    h_distance_target=hamming_distance)
                weight_original = weight_test.to(torch.float)
                weight_new = weight_test_new.to(torch.float)

                weight_original = weight_original.to(device)
                weight_new = weight_new.to(device)
                criterion = nn.MSELoss()
                loss = criterion(weight_original, weight_new)
                #print(f'{format}: MSE loss between new weight and original weight is {loss}')
                print(f'{format.ljust(15)} MSE: {loss}')
                file.writelines(f'{format.ljust(15)} MSE: {loss} \n')
            file.writelines('\n')
        file.close()

           
if __name__ == "__main__":
    main()

                    
                    

            