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
    total_param = 0

    for i in range(len(weight_list)):
        weight_test = weight_list[i]
        size = list(weight_test.size())
        num_param = torch.prod(torch.Tensor(size))
        total_param += num_param
    print(f'Total params:  {total_param} \n')

    for i in range(len(weight_list)):
        weight_test = weight_list[i]
        size = list(weight_test.size())
        num_param = torch.prod(torch.Tensor(size))
        print(f'Layer {name_list[i]}')
        print(f'# of params: {num_param}')
        print(f'% of params: {num_param / total_param}\n')

                    
if __name__ == "__main__":
    main()
