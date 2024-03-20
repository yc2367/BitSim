import torch
import torch.nn as nn
import torchvision
import numpy as np
from util.count_sparsity import *

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

GROUP_SIZE = 4
w_bitwidth = 8

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = countZeroColumn()

    for i in range(1,2):
        weight_test = weight_list[i]
        print(weight_test.shape)
        print(f'Layer {name_list[i]}')

        if len(weight_test.shape) == 4:
            count.count_zero_column_conv(weight_test, w_bitwidth=w_bitwidth, 
                                         group_size=GROUP_SIZE, device=device)
        elif len(weight_test.shape) == 2:
            count.count_zero_column_fc(weight_test, w_bitwidth=w_bitwidth, 
                                       group_size=GROUP_SIZE, device=device)
        
        print(f'# of zero column: {count.num_zero_column}')
        print(f'# of total column: {count.num_total_column}')

                    
if __name__ == "__main__":
    main()
