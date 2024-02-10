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

weight_test = weight_list[13]
GROUP_SIZE = 16

w_bitwidth = 8

def main():
    weight_test_new = process_signMagnitude_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, pruned_column_num=4)
    weight_original = weight_test.to(torch.float)
    weight_new = weight_test_new.to(torch.float)

    criterion = nn.MSELoss()
    loss = criterion(weight_original, weight_new)
    print(f'MSE loss between new weight and original weight is {loss}')

                    
if __name__ == "__main__":
    main()

            
