import torch
import torch.nn as nn
import torchvision
import numpy as np
from util import *

a = torch.Tensor([-15, -9, 23, -14, 18, -2, -12, -6, -14, 7, -14, 18, -30, 6, 10, 4])

group_size = 16
a_q = int_to_twosComplement(a)
a_new = bitFlip_twosComplement(a, a_q, zero_column_required=5)

print(a_new)
