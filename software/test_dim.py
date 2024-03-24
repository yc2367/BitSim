import torch, torchvision
from util.bitflip_layer import *
import argparse

from torchvision.models import resnet50, ResNet50_Weights


model = resnet50(weights = ResNet50_Weights)

model = model.cpu()

weight_list = []
name_list   = []
for n, m in model.named_modules():
    if hasattr(m, "weight"):
        if n == 'fc':
            in_features = m.in_features
            print(in_features)
        w = m.weight
        weight_list.append(w)
        name_list.append(n)
