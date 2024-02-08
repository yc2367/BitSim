import torch
import torch.nn as nn
import torchvision
import numpy as np
from util_old import *

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

weight_test = weight_list[10]
GROUP_SIZE = 16

func = 1
if func == 0:
    convert = lambda t: int_to_sign_magnitude(t, 8)
    file_name = 'resnet18_weight_sign_magnitude.txt'
else: 
    convert = lambda t: int_to_2s_complement(t, 8)
    file_name = 'resnet18_weight_2s_complement.txt'
vconvert = np.vectorize(convert)

def main():
    with open(file_name, 'w') as f:
        for k in range(weight_test.shape[0]):  # output channel
            for x in range(weight_test.shape[3]):  # kernel width
                for y in range(weight_test.shape[2]):  # kernel height
                    for c in range(weight_test.shape[1] // GROUP_SIZE):  # input channel
                        group = weight_test[k, c*GROUP_SIZE:(c+1)*GROUP_SIZE, x, y]
                        group = group.detach().cpu().numpy()
                        group_value = vconvert(group)
                        for v in group_value:
                            f.writelines(v + '\n')
                        f.writelines('\n')
                    
if __name__ == "__main__":
    main()

            
