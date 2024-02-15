"""
Calibrator of post-training quantization (PTQ)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Union
from torch.nn.modules import Module
from tqdm import tqdm

from src.utils.utils import accuracy, AverageMeter
from src.module.convert import get_parent_name
from src.module.base import _QBaseConv2d, _QBaseLinear
from src.module.attention import QAttention

from src.quantization.adaround import AdaRound
from src.quantization.lsq import LSQ

weight_quantizer = {
    "adaround": AdaRound,
}

input_quantizer = {
    "lsq": LSQ
}

def assign_quantizer(model, wbit:int, abit:int, wqtype:str, xqtype:str, train_flag=True):
    """
    Assign the quantizers to CNN all at once
    """
    model = copy.deepcopy(model)
    modules = dict(model.named_modules(remove_duplicate=True))

    for n, m in modules.items():
        if isinstance(m, (_QBaseConv2d, _QBaseLinear)):
            
            m.wq = weight_quantizer[wqtype](nbit=wbit, train_flag=train_flag, weights=m.weight)
            m.aq = input_quantizer[xqtype](nbit=abit, train_flag=train_flag)

            parent_name, name = get_parent_name(n)
            setattr(modules[parent_name], name, m)
    return model

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

class PTQ(object):
    """
    PTQ trainer
    """
    def __init__(self, 
                model: nn.Module, 
                loss_type: str, 
                trainloader, 
                validloader, 
                args, 
                logger):
        # model architecture
        self.model = model

        # args
        self.args = args

        # qtypes
        self.wqtype = self.args.wqtype
        self.xqtype = self.args.xqtype

        # loader
        self.trainloader = trainloader
        self.validloader = validloader

        # max iterations
        self.epochs = self.args.epochs

        # logger
        self.logger = logger
        self.logger_dict = {}

        # loss func
        if loss_type == "mse":
            self.criterion = torch.nn.MSELoss().cuda()
        elif loss_type == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            raise NotImplementedError("Unknown loss type")
        
        # cuda
        self.model = self.model.cuda()
        self.steps = len(self.trainloader)

    def fetch_layer_data(self, layer:nn.Module, batch):
        hook = DataSaverHook(store_input=True, store_output=True)
        handle = layer.register_forward_hook(hook)

        with torch.no_grad():
            out = self.model(batch)
        
        handle.remove()
        return hook.input[0].detach(), hook.output.detach()

    def fetch_layer_data_all(self, layer:nn.Module):
        cached_data = []
        
        pbar = tqdm(self.trainloader, desc="Fetch Data")
        for idx, (inputs, target) in enumerate(pbar):
            inputs = inputs.cuda()
            
            x, y = self.fetch_layer_data(layer, inputs)
            cached_data.append((x, y))
        
        return cached_data    

    def layer_calibrator(self, layer:Union[_QBaseConv2d, _QBaseLinear], cached_data):
        # assign the layer quantizer
        weight = layer.weight
        layer.weight.requires_grad_(False)

        # bias flag
        hasbias = layer.bias is not None
        if hasbias:
            layer.bias.requires_grad_(False)
        
        # quantizer parameters
        qparams = []

        layer.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=weight, train_flag=True).cuda()
        qparams += [
            {'params':layer.wq.parameters(), 'lr': self.args.lr, 'weight_decay': 0.0}, 
        ]

        layer.aq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True).cuda()
        qparams += [
            {'params':layer.aq.parameters(), 'lr': self.args.lr, 'weight_decay': 0.0}, 
        ]

        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(qparams, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(qparams, weight_decay=self.args.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(cached_data)), eta_min=0.)
        pbar = tqdm(range(self.epochs), desc="Epoch")
        for i in pbar:
            calib_loss = AverageMeter()
            for idx, batch in enumerate(cached_data):
                # fetch the data
                x, y = batch

                # cuda
                x = x.cuda()
                y = y.cuda()

                out = layer(x)
                
                err = F.mse_loss(out, y)
                calib_loss.update(err.item())

                optimizer.zero_grad()
                err.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

            pbar.set_postfix(lr=scheduler.get_last_lr()[0], loss=calib_loss.avg)

        return layer, calib_loss.avg
    
    def base_forward(self, inputs, target):
        """
        Foward pass of NN
        """
        out = self.model(inputs)
        loss = F.cross_entropy(out, target)
        return out, loss
    
    def valid_step(self, inputs, target):
        """
        Validation step at each iteration
        """
        out, loss = self.base_forward(inputs, target)
            
        return out, loss

    def valid_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        self.model.eval()
        
        with torch.no_grad():
            for idx, (inputs, target) in enumerate(tqdm(self.validloader)):
                inputs = inputs.cuda()
                target = target.cuda(non_blocking=True)
                
                out, loss = self.valid_step(inputs, target)
                prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

                losses.update(loss.mean().item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
        
        self.logger_dict["valid_loss"] = losses.avg
        self.logger_dict["valid_top1"] = top1.avg
        self.logger_dict["valid_top5"] = top5.avg

    def fit(self):
        modules = dict(self.model.named_modules(remove_duplicate=False))

        for n, m in modules.items():
            if isinstance(m, (_QBaseConv2d, _QBaseLinear)):
                # fetch data
                cached_data = self.fetch_layer_data_all(m)

                self.logger.info(f"Start Calibration of layer: {n}")
                new_layer, calib_err = self.layer_calibrator(m, cached_data)
                self.logger.info(f"Layer {n}: Loss = {calib_err}")

                parent_name, name = get_parent_name(n)
                setattr(modules[parent_name], name, new_layer)


class PTQAttention(PTQ):
    def __init__(self, model: Module, loss_type: str, trainloader, validloader, args, logger):
        super().__init__(model, loss_type, trainloader, validloader, args, logger)
        self.q_layers = ["blocks.10.attn", "blocks.11.attn"]
    
    def update_attn(self, layer:QAttention):
        # low precision qkv
        qkvw = layer.qkv.weight
        projw = layer.proj.weight
        
        # freeze weights
        layer.qkv.weight.requires_grad_(False)
        
        if layer.qkv.bias is not None:
            layer.qkv.bias.requires_grad_(False)

        layer.proj.weight.requires_grad_(False)
        layer.proj.bias.requires_grad_(False)

        # low precision qkv
        layer.qkv.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=qkvw, train_flag=True).cuda()
        layer.xq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True).cuda()
        layer.qqkv = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True).cuda()

        # low precision weights
        layer.proj.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=projw, train_flag=True).cuda()
        layer.proj.aq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True).cuda()
        
        return layer

    def layer_calibrator(self, layer:QAttention, cached_data):
        qlayer = self.update_attn(layer)
        
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(qlayer.parameters(), weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(qlayer.parameters(), weight_decay=self.args.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(cached_data)), eta_min=0.)
        pbar = tqdm(range(self.epochs), desc="Epoch")
        for i in pbar:
            calib_loss = AverageMeter()
            for idx, batch in enumerate(cached_data):
                # fetch the data
                x, y = batch

                # cuda
                x = x.cuda()
                y = y.cuda()

                out = layer(x)
                
                err = F.mse_loss(out, y)
                calib_loss.update(err.item())

                optimizer.zero_grad()
                err.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

            pbar.set_postfix(lr=scheduler.get_last_lr()[0], loss=calib_loss.avg)

        return qlayer, calib_loss.avg

    def fit(self):
        modules = dict(self.model.named_modules(remove_duplicate=False))

        for n, m in modules.items():
            if isinstance(m, QAttention):
                if n in self.q_layers:
                    # fetch data
                    cached_data = self.fetch_layer_data_all(m)

                    self.logger.info(f"Start Calibration of layer: {n}")
                    new_layer, calib_err = self.layer_calibrator(m, cached_data)
                    self.logger.info(f"Layer {n}: Loss = {calib_err}")
                    
                    parent_name, name = get_parent_name(n)
                    setattr(modules[parent_name], name, new_layer)