import math
import torch
import torch.nn as nn
import numpy as np

from typing import List

def count_zero_ops_conv(weight, input, output):
    stride = weight