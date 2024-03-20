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
            sparse_value_count, param_count = count_zero_value_conv(weight_test)
        elif len(weight_test.shape) == 2:
            sparse_value_count, param_count = count_zero_value_fc(weight_test)
        
        print(f'# of zero param: {sparse_value_count}')
        print(f'# of total param: {param_count}')

                    
if __name__ == "__main__":
    main()
