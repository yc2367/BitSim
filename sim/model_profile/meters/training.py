"""
Trainng latency: Forward pass vs. Backward pass
"""

import time
import torch
import torch.nn as nn

from model_profile.meters.profiler import Profiler
from thop import profile

from tqdm import tqdm
from typing import List

class TrainingProfiler(Profiler):
    def __init__(self, name, model: nn.Module, device, input_size: int, precision=32, batch_size:List=[256, 64, 1], runs:int=1):
        super().__init__(name, model, device, input_size, precision)
        
        self.batch_size = batch_size
        self.runs = runs

        # define the loss
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        # define the scaling factor between the forward and backward pass
        # forward and backward latency measured by Pytorch Profile
        self.ratio = 1.958366883

    def cuda_step(self, batch):
        # timer 
        forward_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.runs)]
        forward_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.runs)]

        backward_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.runs)]
        backward_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.runs)]

        # dummy input
        x = torch.randn(batch, 3, self.input_size, self.input_size).cuda()
        target = torch.randn(batch, 1000).cuda(non_blocking=True)

        # warmup the forward pass to avoid the variation of data communication
        print("Quick Warmup...")
        for _ in tqdm(range(1)):
            y = self.model(x)
            loss = self.criterion(y, target)
            loss.backward()

        # delete the variable to aovid the memory error
        del y
        del target
        del x

        x = torch.randn(batch, 3, self.input_size, self.input_size).cuda()
        target = torch.randn(batch, 1000).cuda(non_blocking=True)

        for i in tqdm(range(self.runs)):
            forward_start_events[i].record()
            y = self.model(x)
            loss = self.criterion(y, target)
            forward_end_events[i].record()

            backward_start_events[i].record()
            self.optimizer.zero_grad()
            loss.backward()
            backward_end_events[i].record()

            self.optimizer.step()
            
            torch.cuda.empty_cache()

        # wait for gpu to synchornize
        torch.cuda.synchronize()
        
        forward_times = [s.elapsed_time(e) for s, e in zip(forward_start_events, forward_end_events)]
        backward_times = [s.elapsed_time(e) for s, e in zip(backward_start_events, backward_end_events)]
        
        return sum(forward_times) / self.runs, sum(backward_times) / self.runs
    
    def batch_sweep(self):
        fwd_time_dict = {}
        bwd_time_dict = {}
        ops_dict = {}
        
        fwd_time_dict["name"] = self.name
        bwd_time_dict["name"] = self.name
        ops_dict["name"] = self.name
        for b in self.batch_size:
            
            fwd, bwd = self.cuda_step(b)
            unit = "ms"

            # profile macs
            macs, params = self.step()
            
            fwd_time_dict["b="+str(b)+f"({unit})"] = fwd
            bwd_time_dict["b="+str(b)+f"({unit})"] = bwd

            ops_dict["macs_fwd"] = macs
            ops_dict["macs_bwd"] = macs * self.ratio
            ops_dict["flop_fwd"] = macs * 2
            ops_dict["flop_bwd"] = macs * self.ratio * 2
        
        return fwd_time_dict, bwd_time_dict, ops_dict