import torch
import torch.nn as nn
import torchvision
import numpy as np
from util_profile import *

from torchvision.models.quantization import ResNet18_QuantizedWeights
model = torchvision.models.quantization.resnet18(weights = ResNet18_QuantizedWeights, quantize=True)

model = model.cpu()


count = 0
layer_num = 5
for n, m in model.named_modules():
    if hasattr(m, "weight"):
        print(m)

print('\n')
for n, m in model.named_modules():
    if hasattr(m, "weight"):
        if count == layer_num:
            w = m.weight()
            print(m)
            wint = torch.int_repr(w)
            weight_test = wint
        count += 1

print(weight_test.shape)
CHANNEL_GROUP_SIZE = 1

func = 0
if func == 0:
    convert = lambda t: int_to_sign_magnitude(t, 8)
    file_name = 'resnet18_weight_sign_magnitude.txt'
else: 
    convert = lambda t: int_to_2s_complement(t, 8)
    file_name = 'resnet18_weight_2s_complement.txt'
vconvert = np.vectorize(convert)

def main():
    with open(file_name, 'w') as f:
        g_count_1 = 0
        for kt in range(weight_test.shape[0] // CHANNEL_GROUP_SIZE):  # output channel
            count_1 = 0
            count_2 = 0
            for k in range(CHANNEL_GROUP_SIZE):
                #count_2 = 0
                group = weight_test[kt*CHANNEL_GROUP_SIZE+k, :, :, :]
                group = group.reshape(-1).detach().cpu().numpy()
                group_value = vconvert(group)
                for v in group_value:
                    if v[3] == '1':
                        count_1 += 1
                    else:
                        if (v[3] == '0') and (v[6] == '1'):
                            count_2 += 1
                    #if v[6] == '1':
                        #count_2 += 1
                    f.writelines(v + '\n')
                #if count_2 > 4:
                    #g_count_2 += 1
            if count_1 < (weight_test.shape[3] * weight_test.shape[2] * weight_test.shape[1] / 32):
                g_count_1 += 1
            s = 'output channel group ' + str(kt) + ' at location '
            f.writelines(s + '\n')
            f.writelines(f'count 1st MSB: {count_1} \n')
            f.writelines(f'count 2nd MSB: {count_2} \n')
            f.writelines(f'count total: {weight_test.shape[3] * weight_test.shape[2] * weight_test.shape[1]} \n')
            f.writelines('\n\n')
        f.writelines(f'count MSB grounp size: {g_count_1} \n')
        f.writelines('\n\n')
        
                    
if __name__ == "__main__":
    main()

            
