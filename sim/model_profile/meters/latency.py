"""
Batch size vs. Latency profiler
"""

import time
import torch
import torch.nn as nn

from model_profile.meters.profiler import Profiler
from model_profile.meters.utils import AverageMeter

from tqdm import tqdm
from typing import List

class BatchProfiler(Profiler):
    def __init__(self, name, model: nn.Module, device, input_size: int, precision=32, batch_size:List=[1,64,256], runs:int=1):
        super().__init__(name, model, device, input_size, precision)
        self.batch_size = batch_size
        self.runs = runs

    def cuda_forward(self, batch):
        # timer 
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.runs)]

        # dummy input
        x = torch.randn(batch, 3, self.input_size, self.input_size).cuda()

        # warmup the forward pass to avoid the variation of data communication
        print("Quick Warmup...")
        for _ in tqdm(range(1)):
            y = self.model(x)

        # delete the variable to aovid the memory error
        del y
        del x

        # reintialize the tensor
        x = torch.randn(batch, 3, self.input_size, self.input_size).cuda()

        for i in tqdm(range(self.runs)):
            start_events[i].record()
            _ = self.model(x)
            
            end_events[i].record()
            
            torch.cuda.empty_cache()

        # wait for gpu to synchornize
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        return sum(times) / self.runs
    
    def cpu_forward(self, batch):
        # dummy input
        x = torch.randn(batch, 3, self.input_size, self.input_size).cpu()

        # warmup the forward pass to avoid the variation of data communication
        print("Quick Warmup...")
        for _ in tqdm(range(1)):
            y = self.model(x)

        # delete the variable to aovid the memory error
        del y
        del x

        # reintialize the tensor
        x = torch.randn(batch, 3, self.input_size, self.input_size).cpu()

        meter = AverageMeter()
        for i in tqdm(range(self.runs)):
            start = time.time()
            y = self.model(x)
            duration = time.time() - start

            meter.update(duration)
        
        return meter.avg
    
    def batch_sweep(self):
        time_dict = {}
        
        time_dict["name"] = self.name
        for b in self.batch_size:
            if self.device == "cuda":
                rt = self.cuda_forward(b)
                unit = "ms"
            elif self.device == "cpu":
                rt = self.cpu_forward(b)
                unit = "s"

            time_dict["b="+str(b)+f"({unit})"] = rt
        
        return time_dict
    


        

        

        

        