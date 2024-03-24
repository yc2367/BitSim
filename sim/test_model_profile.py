from model_profile.models.models import MODEL
from model_profile.meters.dim import DIM
import torch.nn as nn

model = MODEL['resnet50']
model = model()
profiler = DIM('resnet50', model, device="cuda", input_size=224)

profiler.fit()

for n, m in model.named_modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        print(n)
        print(profiler.w_dict[n])
        print(profiler.i_dict[n])
        print(profiler.o_dict[n])
        print()
