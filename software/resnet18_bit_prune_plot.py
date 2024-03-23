import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from util.bitflip_layer import *

import warnings 
warnings.filterwarnings("ignore")

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

loss = 1
if loss == 0:
    metric = 'MSE'
else: 
    metric = 'KL_DIV'

pruned_col_num = 2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for N in range(pruned_col_num, pruned_col_num+1):
        num_pruned_column = N
        file = open(f'resnet18_prune_loss_report_g{GROUP_SIZE}_c{num_pruned_column}.txt', 'w')

        for i in range(1, len(weight_list)):
            weight_test = weight_list[i]
            print(f'Layer {name_list[i]}')
            file.writelines(f'Layer {name_list[i]} \n')

            #print(weight_test.unique())
            plt.figure(dpi=300)
            f = sns.displot(weight_test.cpu().reshape(-1).numpy())
            f.fig.suptitle(str(i) + '  ' +str(name_list[i]))
            f.savefig(f'./plot/{name_list[i]}_original.png')
            
            #print([weight_test[weight_test.eq(i)].numel() for i in range(-40, -20)])
            for func in [0, 1, 2, 3]:
                if func == 0:
                    format = 'Round to Nearest'
                    if len(weight_test.shape) == 4:
                        weight_test_new = bitflip_signMagnitude_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    num_pruned_column=num_pruned_column, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = bitflip_signMagnitude_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                num_pruned_column=num_pruned_column, device=device)
                elif func == 1:
                    format = 'Column Averaging'
                    if len(weight_test.shape) == 4:
                        weight_test_new = colAvg_twosComplement_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    num_pruned_column=num_pruned_column, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = colAvg_twosComplement_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    num_pruned_column=num_pruned_column, device=device)        
                elif func == 2:
                    format = 'Zero Point'
                    if len(weight_test.shape) == 4:
                        weight_test_new = bitflip_zeroPoint_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    num_pruned_column=num_pruned_column, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = bitflip_zeroPoint_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    num_pruned_column=num_pruned_column, device=device)
                else:
                    format = 'Optimal'
                    if len(weight_test.shape) == 4:
                        weight_test_new = bitVert_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                        num_pruned_column=num_pruned_column, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = bitVert_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                    num_pruned_column=num_pruned_column, device=device)
                #print(weight_test_new.unique())
                
                # plot distribution
                plt.figure(dpi=300)
                f = sns.displot(weight_test_new.cpu().reshape(-1).numpy())
                f.fig.suptitle(str(i) + '  ' +str(name_list[i]))
                f.savefig(f'./plot/{name_list[i]}_new_{format.replace(" ", "")}.png')

                weight_original = weight_test.to(dtype=torch.float, device=device)
                weight_new = weight_test_new.to(device)

                if metric == 'MSE':
                    criterion = nn.MSELoss()
                    loss = criterion(weight_original, weight_new)
                else:
                    criterion = nn.KLDivLoss();
                    weight_original = F.log_softmax(weight_original.reshape(-1), dim=0)
                    weight_new = F.softmax(weight_new.reshape(-1), dim=0)
                    loss = criterion(weight_original, weight_new)

                #print(f'{format}: MSE loss between new weight and original weight is {loss}')
                print(f'{format.ljust(20)} {metric}: {loss}')
                file.writelines(f'{format.ljust(20)} {metric}: {loss} \n')
                #print([weight_test_new[weight_test_new.eq(i)].numel() for i in range(-128, 127)])
            print()
            file.writelines('\n')
        file.close()

                    
if __name__ == "__main__":
    main()
