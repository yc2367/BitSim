import torch
from util import *

# integer weights (torch.FloatTensor)
wq = torch.Tensor([-15, -9, 23, -14, 18, -2, -12, -6, -14, 7, -14, 18, -30, 6, 10, 4])
w_bitwidth = 8

# int to sign magnitude
func = 0
if func == 0:
    wqb_list = int_to_signMagnitude(wq, w_bitwidth=w_bitwidth, cellBit=1)
    wq_original_list = signMagnitude_to_int(wqb_list, w_bitwidth=w_bitwidth)
else:
    wqb_list = int_to_twosComplement(wq, w_bitwidth=w_bitwidth, cellBit=1)
    wq_original_list = twosComplement_to_int(wqb_list, w_bitwidth=w_bitwidth)

wb = wqb_list.contiguous().view(8, -1)
wq_original = wq_original_list.view(-1)

for i in range(len(wq)):
    wbi = wb[:, i]
    wqi = wq[i]
    wqoi = wq_original[i]

    print(wqi)
    print(wbi)
    print(wqoi)

    print("====")
    
print("Note: Right most = MSB")

wqb_new = bitFlip_signMagnitude(wq, w_bitwidth=8, zero_column_required=5)
print(wqb_new.shape)
wq_new = signMagnitude_to_int(wqb_new, w_bitwidth=8)
print(wq_new.shape)

for i in range(len(wq)):
    wbi = wqb_new[:, i]
    wqi = wq_new[i]
    wqoi = wq_original[i]

    print(wqoi)
    print(wbi)
    print(wqi)

    print("====")

print(wq_original)

print(wq_new)
    