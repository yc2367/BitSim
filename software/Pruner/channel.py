"""
Pruner for bidirectional pruning
"""

import json
import torch
import torch.nn as nn

from src.module.base import _QBase, _QBaseConv2d, _QBaseLinear
from src.module.fuse import QConvReLU, QConvBNReLU

class Pruner(object):
    def __init__(self, model:nn.Module, pr:float=0.5):
        self.model = model
        self.masks = {}

        # pruning ratio
        self.pr = pr

        # group_size
        self.grp = 32
    
    def overall_sparsity(self):
        nz = 0
        total = 0
        for k, v in self.masks.items():
            total += v.numel()
            nz += v.sum()

        return 1 - nz / total

    def magnitude(self, scale:torch.Tensor):
        pass

    def rel_magnitude(self, scale:torch.Tensor):
        pass

    def collect_score(self):
        channel_scores = []
        for n, m in self.model.named_modules():
            if isinstance(m, _QBase):
                if ".wq" in n:
                    delta = 1 / m.scale.data.view(-1).abs()
                    channel_scores.append(delta)
        
        channel_scores = torch.cat([torch.flatten(x) for x in channel_scores])
        return channel_scores
    
    def get_threshold(self):
        # get scores
        mp_scores = self.collect_score()
        num_params_to_keep = int(len(mp_scores) * (1 - self.pr))
        topkscore, _ = torch.topk(mp_scores, num_params_to_keep, sorted=True)
        return topkscore[-1]

    def structured(self, mask:torch.Tensor, delta:torch.Tensor):
        nz = mask.sum()

        if nz.item() % self.grp == 0:
            return mask
        else:
            m = nz.div(self.grp).round().mul(self.grp)
            
            if m == 0:
                m = nz.div(self.grp).ceil().mul(self.grp)
            
            topkscore, _ = torch.topk(delta, int(m), sorted=True)
            
            try:
                if self.pr < 1.0:
                    threshold = topkscore[-1]
                else:
                    threshold = torch.tensor('inf')
            except:
                import pdb;pdb.set_trace()

            mask = delta.ge(threshold).float()
        return mask

    def update_mask(self, threshold):
        for n, m in self.model.named_modules():
            if isinstance(m, (QConvBNReLU, QConvReLU)):
                mask = torch.zeros(m.conv.out_channels, device=m.conv.weight.device)
                scale = m.conv.wq.scale.data.abs().view(-1)
                delta = 1 / scale

                if len(scale) > 1:
                    mask[delta.gt(threshold)] = 1.0

                # if mask.sum() < len(delta):
                #     mask = self.structured(mask, delta)
                
                self.masks[n] = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                print(f"{n} | Dense channels = {int(mask.sum().item())}")
    
    def export(self):
        nz_indexes = {}
        for k, v in self.masks.items():
            if len(v.size()) == 4:
                v = v.squeeze(1).squeeze(1).squeeze(1)
            elif len(v.size()) == 2:
                v = v.squeeze(1)

            idx = (v != 0).nonzero().squeeze(1)
            nz_indexes[k] = idx.tolist()

        return nz_indexes

    def step(self):
        if self.pr < 1.0:
            threshold = self.get_threshold()
        else:
            threshold = torch.tensor(1e+5)
        self.update_mask(threshold)
