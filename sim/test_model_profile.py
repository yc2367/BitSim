from model_profile.models.models import MODEL
from model_profile.meters.dim import DIM
import torch.nn as nn
import torchvision
import numpy as np

model = MODEL['resnet50']
model = model()
profiler = DIM('resnet50', model, device="cuda", input_size=224)

profiler.fit()

for n, m in model.named_modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, torchvision.models.resnet.Bottleneck):
        print(n)
        print(profiler.weight_dim[n])
        print(profiler.input_dim[n])
        print(profiler.output_dim[n])
        if profiler.weight_dim[n] is not None:
            print(f'weight mem (KB): {np.prod(profiler.weight_dim[n]) / 1024}')
            print(f'input mem (KB): {np.prod(profiler.input_dim[n]) / 1024}')
        print()
