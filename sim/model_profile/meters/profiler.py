"""
DNN Profiler
"""

import torch
import torch.nn as nn
from thop import profile

MB = 1024.0 * 1024.0
BYTE = 8

def hook(name, state_dict):
    def hook(model, input, output):
        state_dict[name] = output.detach().numel()
    return hook

class Profiler(object):
    def __init__(self, model_name, model:nn.Module, device, input_size:int, precision=32) -> None:
        self.input_size = input_size
        self.device = device
        self.precision = precision
        self.model_name = model_name
        
        if device == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available"
            self.model = model.cuda()
            self.device_name = torch.cuda.get_device_name(0)
        elif device == "cpu":
            self.device_name = "cpu"
            self.model = model.cpu()
        else:
            raise ValueError("Device type can only be 'cpu' or 'cuda'!")
        
        self.logger_dict = {}
        self.feature_dict = {}
        self.info = {}

    @property
    def _info(self):
        return self.info

    def count2mb(self, count):
        mem = count*self.precision / BYTE / 1e+6
        return mem
    
    def step(self):
        x = torch.randn(1, 3, self.input_size, self.input_size)
        x = x.cuda() if self.device == "cuda" else x

        macs, params = profile(self.model, inputs=(x, ))
        return macs, params
    
    def hook(self):
        for n, m in self.model.named_modules():
            m.register_forward_hook(hook(n, self.feature_dict))

    def forward(self):
        # hook
        self.hook()
        
        x = torch.randn(1, 3, self.input_size, self.input_size)
        x = x.cuda() if self.device == "cuda" else x
        
        # forward pass 
        with torch.no_grad():
            y = self.model(x)

    def feature_mem(self):
        feature_count = 0

        for k, v in self.feature_dict.items():
            feature_count += v
        
        mb = self.count2mb(feature_count)
        return mb

    def profile(self):
        with torch.no_grad():
            macs, params = self.step()
            wmb = self.count2mb(params)

            # dry run for activation stats
            self.forward()

        # compute the memory size of activation
        fmb = self.feature_mem()

        # memory footprint
        mbf = wmb + fmb
        
        # read from device
        memory = torch.cuda.max_memory_allocated() / MB
        print(f"Device = {self.device_name}")
        print(f"MACS = {macs}")
        print(f"Params = {params}")
        print(f"Memory usage (READ from GPU) = {memory} MB")

        # update the info
        self.info["name"] = self.model_name
        self.info["device"] = self.device_name
        self.info["macs"] = macs
        self.info["flops"] = macs * 2
        self.info["params"] = params
        self.info["gpu_memory"] = memory
        self.info["weight_mb"] = wmb
        self.info["feature_mb"] = fmb
        self.info["mem_footprint"] = mbf
        self.info["ops_intensity_gpu"] = macs * 2 / (memory * 1e+6)
        self.info["ops_intensity_cal"] = macs * 2 / (mbf * 1e+6)

        print(f"Memory usage (calculated) = {mbf} MB")
