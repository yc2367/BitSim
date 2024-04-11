import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from util.bitflip_layer import *

import warnings 
warnings.filterwarnings("ignore")

from torchvision.models.quantization import (ResNet18_QuantizedWeights, 
                                             MobileNet_V2_QuantizedWeights,
                                             ResNet50_QuantizedWeights)

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices = ['resnet18', 'resnet50', 'mobilenet'])
args = parser.parse_args()

model_name = args.model

if model_name == 'resnet18':
    weights = ResNet18_QuantizedWeights
    model = torchvision.models.quantization.resnet18(weights = weights, quantize=True)
elif model_name == 'resnet50':
    weights = ResNet50_QuantizedWeights
    model = torchvision.models.quantization.resnet50(weights = weights, quantize=True)
elif model_name == 'mobilenet':
    weights = MobileNet_V2_QuantizedWeights
    model = torchvision.models.quantization.mobilenet_v2(weights = weights, quantize=True)
else:
    raise ValueError('ERROR! The provided model is not one of supported models.')

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

pruned_col_num = 3

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_title = ['ResNet50 Conv4.1.3 Weight', 'With BitWave Bit-flip', 'With BitVert Binary-Pruning']
    color_list = [(140/255, 205/255, 140/255), (255/255, 150/255, 150/255), (140/255, 205/255, 140/255)]
    ticksize = 14
    for N in range(pruned_col_num, pruned_col_num+1):
        num_pruned_column = N
        file = open(f'{model_name}_prune_loss_report_g{GROUP_SIZE}_c{num_pruned_column}.txt', 'w')
        for i in range(len(weight_list)-5, len(weight_list)-4):
            weight_test = weight_list[i]
            weight_shape = weight_list[i].shape
            print(f'Layer {name_list[i]}')
            print(f'Layer Shape: {weight_shape}')
            file.writelines(f'Layer {name_list[i]} \n')
            file.writelines(f'Layer Shape: {weight_shape} \n')

            for func in [0, 1, 2,]:
                if func == 0:
                    format = 'Original'
                    weight_test_new = weight_test
                    weight_test_new_5b = torch.load('layer4.1.conv3.conv.ops_x2_baseline5bit.pt')
                if func == 1:
                    format = 'Round to Nearest'
                    if len(weight_test.shape) == 4:
                        weight_test_new = bitflip_signMagnitude_conv(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
                                                                    num_pruned_column=num_pruned_column, device=device)
                    elif len(weight_test.shape) == 2:
                        weight_test_new = bitflip_signMagnitude_fc(weight_test, w_bitwidth=w_bitwidth, group_size=GROUP_SIZE, 
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
                    pass
                
                # plot distribution
                color = color_list[func]
                
                if func == 1:
                    num_bin = 64
                else:
                    num_bin = 64
                fig, ax = plt.subplots()
                fig.set_figwidth(4)
                fig.set_figheight(5)
                sns.histplot(weight_test_new.cpu().reshape(-1).numpy(), 
                            bins=num_bin, color=color, ec=None, ax=ax)
                if func == 0:
                    sns.histplot(weight_test_new_5b.cpu().reshape(-1).numpy(), 
                            bins=32, color=(255/255, 150/255, 150/255), ec=None, ax=ax)
                ax.set_xlabel('Weight Value', fontsize=ticksize+1, weight='bold', labelpad=5)
                if func == 0:
                    ax.set_ylabel(r'Count ($\times$1000)', fontsize=ticksize+1, weight='bold', labelpad=0)
                else:
                    ax.set_ylabel(r'   ', fontsize=ticksize+1, weight='bold', labelpad=0)
                ax.set_xlim(-135, 135)
                ax.set_ylim(0, 1e5)
                x_ticklabels = ax.get_xticks()
                y_ticklabels = ax.get_yticks()
                x_ticklabels = [int(n) for n in x_ticklabels]
                y_ticklabels = [int(n // 1e3) for n in y_ticklabels]
                if func == 0:
                    ax.tick_params('y', length=5)
                else:
                    ax.tick_params('y', length=0)
                ax.set_yticklabels(y_ticklabels, fontsize=ticksize)
                ax.set_xticklabels(x_ticklabels, fontsize=ticksize)
                
                #ax.set_title(fig_title[func], fontsize=ticksize, weight='bold')
                fig.savefig(f'./plot/{name_list[i]}_new_{format.replace(" ", "")}.png', dpi=200, bbox_inches="tight")

                weight_original = weight_test.to(dtype=torch.float, device=device)
                weight_new = weight_test_new.to(device)

                if func != 0:
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
            file.writelines('\n')
            print()
        file.close()

                    
if __name__ == "__main__":
    main()
