import torch
import torch.nn as nn
import torchvision
import numpy as np
from util import *

from torchvision.models.quantization import ResNet18_QuantizedWeights
model = torchvision.models.quantization.resnet18(weights = ResNet18_QuantizedWeights, quantize=True)

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
    pruned_column_num = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, weight_test in enumerate(weight_list):
        print(f'Layer {name_list[i]}')
        for func in [0, 1]:
            if func == 0:
                format = 'Sign Magnitude'
                if len(weight_test.shape) == 4:
                    weight_test_new = process_signMagnitude_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                 pruned_column_num=pruned_column_num)
                elif len(weight_test.shape) == 2:
                    weight_test_new = process_signMagnitude_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                               pruned_column_num=pruned_column_num)
            else:
                format = '2s Complement'
                if len(weight_test.shape) == 4:
                    weight_test_new = process_twosComplement_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                  pruned_column_num=pruned_column_num)
                elif len(weight_test.shape) == 2:
                    weight_test_new = process_twosComplement_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                pruned_column_num=pruned_column_num)
            weight_original = weight_test.to(torch.float)
            weight_new = weight_test_new.to(torch.float)

            weight_original = weight_original.to(device)
            weight_new = weight_new.to(device)
            criterion = nn.MSELoss()
            loss = criterion(weight_original, weight_new)
            #print(f'{format}: MSE loss between new weight and original weight is {loss}')
            print(f'{format.ljust(15)} MSE: {loss}')


                    
if __name__ == "__main__":
    main()

                    
                    

            
