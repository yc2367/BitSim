import torch
import torch.nn as nn
import torchvision
import numpy as np
from util_profile import *

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

weight_test = weight_list[4]
print(weight_test.shape)
GROUP_SIZE = 16
CHUNK_SIZE = 64

func = 0
if func == 0:
    convert = lambda t: int_to_sign_magnitude(t, 8)
    file_name = 'resnet50_weight_sign_magnitude.txt'
else: 
    convert = lambda t: int_to_2s_complement(t, 8)
    file_name = 'resnet50_weight_2s_complement.txt'
vconvert = np.vectorize(convert)

def main():    
    with open(file_name, 'w') as f:
        group_count_1_list = []
        num_chunk = 0
        for c in range(weight_test.shape[1] // GROUP_SIZE):  # input channel
            for x in range(weight_test.shape[3]):  # kernel width
                for y in range(weight_test.shape[2]):  # kernel height
                    for kt in range(weight_test.shape[0] // CHUNK_SIZE):  # output channel
                        num_chunk += 1
                        g_count_1 = 0
                        g_count_2 = 0
                        s = 'input channel ' + str(c) + ' output channel ' + str(kt) + ' at location ' + f'({x}, {y})'
                        f.writelines(s + '\n\n')
                        for k in range(CHUNK_SIZE):
                            count_1 = 0
                            count_2 = 0
                            group = weight_test[kt*CHUNK_SIZE+k, c*GROUP_SIZE:(c+1)*GROUP_SIZE, x, y]
                            group = group.detach().cpu().numpy()
                            group_value = vconvert(group)
                            for v in group_value:
                                if v[3] == '1':
                                    count_1 += 1
                                if v[6] == '1':
                                    count_2 += 1
                                f.writelines(v + '\n')
                            if count_1 > 2:
                                g_count_1 += 1
                            if count_2 > 6:
                                g_count_2 += 1
                            f.writelines(f'{count_1} \n')
                            f.writelines('\n')
                        group_count_1_list.append(g_count_1)
                        f.writelines(f'\n{g_count_1} \n{g_count_2} \n')
    group_count_1_list = np.array(group_count_1_list)
    print(np.sum(group_count_1_list > 16) / num_chunk)

if __name__ == "__main__":
    main()                  

            
