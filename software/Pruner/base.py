"""
Decimal to binary
"""
import time
import os
import json
import sys
sys.path.append("../../")
import torch.nn as nn
import pandas as pd

from src.module.fuse import QConvReLU, QConvBNReLU
from src.module.base import _QBaseLinear
from src.pruner.channel import Pruner


from software.util.bitflip_layer import roundAvg_conv, roundAvg_fc, zeroPointShifting_conv, zeroPointShifting_fc, bitFlip_conv, bitFlip_fc

class D2C(object):
    def __init__(self, model:nn.Module, wbit:int, args):
        self.model = model
        self.wbit = wbit
        self.grp_size = args.grp_size
        self.save_path = args.save_path

        self.pruned_column_num = args.N
        self.func = args.flag   # 0 for signed magnitude, 1 for 2s complement

        # channel wise pruner
        self.pruner = Pruner(model=self.model, pr=args.prune_ratio)
        print(f"Global Prune Ratio = {self.pruner.pr}")


    def fetch_weights(self):
        self.weight_dict = {}

        for n, m in self.model.named_modules():
            if isinstance(m, (QConvBNReLU, QConvReLU)):
                if m.conv.groups == 1:
                    self.weight_dict[n] = m.conv.qweight.data.detach()
            elif isinstance(m, _QBaseLinear):
                self.weight_dict[n] = m.qweight.data.detach()

    def convert(self):
        self.pruner.step()
        self.fetch_weights()

        self.new_weights = {}
        self.prune_ratio = {}
        for k, weight in self.weight_dict.items():
            print(f"Layer [{k}] Start Conversion!")
            if self.func == 0:
                if len(weight.shape) == 4:
                    weight_new = bitFlip_conv(
                        weight, w_bitwidth=self.wbit, group_size=self.grp_size, 
                        num_pruned_column=self.pruned_column_num, device="cuda"
                    )

                if len(weight.shape) == 2:
                    weight_new = bitFlip_fc(
                        weight, w_bitwidth=self.wbit, group_size=self.grp_size, 
                        num_pruned_column=self.pruned_column_num, device="cuda"
                    )


            elif self.func == 1:
                if len(weight.shape) == 4:
                    weight_new = roundAvg_conv(
                        weight, w_bitwidth=self.wbit, group_size=self.grp_size, 
                        num_pruned_column=self.pruned_column_num, device="cuda"
                    )

                if len(weight.shape) == 2:
                    weight_new = roundAvg_fc(
                        weight, w_bitwidth=self.wbit, group_size=self.grp_size, 
                        num_pruned_column=self.pruned_column_num, device="cuda"
                    )

            else:
                if len(weight.shape) == 4:
                    if weight.size(1) != 3:
                        weight_new = zeroPointShifting_conv(
                            weight, w_bitwidth=self.wbit, group_size=self.grp_size, 
                            num_pruned_column=self.pruned_column_num, device="cuda"
                        )
                    else:
                        weight_new = weight

                if len(weight.shape) == 2:
                    weight_new = zeroPointShifting_fc(
                        weight, w_bitwidth=self.wbit, group_size=self.grp_size, 
                        num_pruned_column=self.pruned_column_num, const_bitwidth=4, device="cuda"
                    )
                    

            print(f"Total allocated memory = {torch.cuda.memory_allocated(0)/1e+9:.3f}GB")

            
            if k in self.pruner.masks.keys():
                mask = self.pruner.masks[k]
                weight_mix = weight.mul(mask) + weight_new.mul(1 - mask)

                # bidirectional pruning ratio
                pr = mask.eq(0.0).float().sum().div(mask.numel())
            else:
                print(f"Skip conversion of layer {k}")
                weight_mix = weight
                pr = torch.tensor(0.0)

            error = torch.nn.functional.mse_loss(weight_mix, weight)
            self.new_weights[k] = weight_mix
            print(f"Prune Ratio = {pr} MSE = {error.item():.2f}")
            
            # dump the dict
            nz_dict = self.pruner.export()
            json_path = os.path.join(self.save_path, "nonzero_channels.json")
            with open(json_path, "w") as outfile: 
                json.dump(nz_dict, outfile)
            
            # log
            self.prune_ratio[k] = pr.item()

        print("Conversion completed!")
        df = pd.DataFrame.from_dict(self.prune_ratio, orient="index")
        df.to_csv(os.path.join(self.save_path, "layer_wise_pr.csv"))
        print("DataFrame saved!")

        overall_sparsity = self.pruner.overall_sparsity()
        print(f"Overall sparsity = {overall_sparsity:.3f}")
    
    def reload(self):
        for n, m in self.model.named_modules():
            if isinstance(m, (QConvBNReLU, QConvReLU)):
                if m.conv.groups == 1:
                    m.conv.qweight.data.copy_(self.new_weights[n])
            
            elif isinstance(m, _QBaseLinear):
                m.qweight.data.copy_(self.new_weights[n])
        
        print("Reload completed!")
        return self.model

    def fit(self):
        start = time.time()
        self.convert()
        model = self.reload()
        total = time.time() - start
        print(f"Total time of conversion = {total}s")
        return model