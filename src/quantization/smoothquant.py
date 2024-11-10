"""
T2C Version of SmoothQuant
"""

import torch
import torch.nn as nn
from src.module.base import _QBase

class SmoothQuant(_QBase):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True):
        super().__init__(nbit, train_flag, unsigned)
