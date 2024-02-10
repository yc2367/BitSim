import torch
import torch.nn as nn
import torchvision
from util import *

from torchvision.models.quantization import ResNet50_QuantizedWeights
model = torchvision.models.quantization.resnet50(weights = ResNet50_QuantizedWeights, quantize=True)


for m in model.modules():
    if hasattr(m, "weight"):
        w = m.weight()
        wint = torch.int_repr(w)
        break

# integer weights (torch.FloatTensor)
wq = wint.float()
w_bitwidth = 8

# wq[is_neg] = wq[is_neg].add(2 ** (w_bitwidth - 1))

# int to sign magnitude
func = 0
if func == 0:
    wqb_list = int_to_signMagnitude(wq, w_bitwidth=w_bitwidth, cellBit=1)
    wq_original_list = signMagnitude_to_int(wqb_list, w_bitwidth=w_bitwidth)
else:
    wqb_list = int_to_twosComplement(wq, w_bitwidth=w_bitwidth, cellBit=1)
    wq_original_list = twosComplement_to_int(wqb_list, w_bitwidth=w_bitwidth)


wq = wint[0,1].view(-1)
wb = wqb_list[:,0,1].contiguous().view(8, -1)
wq_original = wq_original_list[0, 1].view(-1)

for i in range(len(wq)):
    wbi = wb[:, i]
    wqi = wq[i]
    wqoi = wq_original[i]

    print(wqi)
    print(wbi)
    print(wqoi)

    print("====")
    
print("Note: Right most = MSB")