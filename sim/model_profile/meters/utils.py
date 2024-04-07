import torch
from typing import Any, Union

class DataSaverHook:
    def __init__(self, store_input=False, store_output=False) -> None:
        self.store_input = store_input
        self.store_output = store_output

        self.input = None
        self.output = None
    
    def __call__(self, module, input_batch, output_batch) -> Any:
        if self.store_input:
            self.input = input_batch
        
        if self.store_output:
            self.output = output_batch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_adap_avgpool(x, y):
    kernel = torch.div(
        torch.DoubleTensor([*(x[0].shape[2:])]), 
        torch.DoubleTensor([*(y.shape[2:])])
    )
    total_add = torch.prod(kernel)
    num_elements = y.numel()
    total_ops = calculate_adaptive_avg(total_add, num_elements)
    return total_ops

def calculate_adaptive_avg(kernel_size, output_size):
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_op * output_size)])
    

