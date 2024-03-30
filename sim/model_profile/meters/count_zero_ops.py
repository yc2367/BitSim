"""
weight & input/output zero operations profiler
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets

from model_profile.meters.profiler import Profiler
from sim.util.model_quantized import MODEL

def get_loader(args):
    # Preparing data
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # read the dataset
    testset = datasets.ImageFolder(args.val_dir, transform=transform_val)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    
    num_classes = 1000
    img_size = 224
    return testloader, num_classes, img_size


def feature_hook(name, state_dict):
    def hook(model, input, output):
        state_dict[name] = [input[0].detach(), output.detach()]
    return hook


class CountZeroOps(Profiler):
    def __init__(self, name, model: nn.Module, args, device, precision=32) -> None:
        (self.testloader, 
         self.num_classes, 
         self.input_size) = get_loader(args)
        super().__init__(name, model, device, self.input_size, precision) 
        self.model_q = MODEL[name].cpu() # quantized model
         
        self.layer_name_list = []
        self.num_zero_input = {} 
        self.num_zero_output = {} 
        self.num_zero_weight = {} 
        # number of zero operations for every layer
        self.num_zero_ops = {}

    def hook(self):
        for n, m in self.model.named_modules():
            m.register_forward_hook(feature_hook(n, self.feature_dict))
    
    def forward(self):
        # hook
        self.hook()
        for (inputs, _) in self.testloader:
            inputs = inputs.cuda() if self.device == "cuda" else x
            # forward pass 
            with torch.no_grad():
                y = self.model(inputs)
            break

    def fit(self):
        self.forward()
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.count_zero_ops_conv(m ,n)
                self.layer_name_list.append(n)
            if isinstance(m, nn.Linear):
                self.count_zero_ops_linear(m, n)
                self.layer_name_list.append(n)

    def count_zero_ops_conv(self, layer: nn.Conv2d, name:str):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        weight = layer.weight
        weight_quantized = self._get_quantized_weight(name)
        
        # size
        bi, cin,  ih, iw = i_feature.shape
        bo, cout, oh, ow = o_feature.shape
        assert bi == bo, 'ERROR! Batch size is not the same for input feature and output feature!'
        # kernel
        cin = layer.in_channels // layer.groups

        self.num_zero_input[name]  = round(torch.sum(i_feature == 0).item() / bi)
        self.num_zero_output[name] = round(torch.sum(o_feature == 0).item() / bi)
        self.num_zero_weight[name] = round(torch.sum(weight_quantized == 0).item())
        print(self.num_zero_input[name])
        print(self.num_zero_output[name])
        print(self.num_zero_weight[name])

        # count number of zero operations for every output pixel
        num_zero_ops = torch.zeros_like(o_feature)
        if cin == layer.in_channels: # conv2D
            unfold = nn.Unfold(kernel_size=layer.kernel_size, padding=layer.padding, stride=layer.stride)
            i_feature = unfold(i_feature)
            for j_cout in range(cout): # output channel dimension
                kernel = weight[j_cout].flatten().unsqueeze(-1).unsqueeze(0).expand(bi, -1, oh*ow)
                is_zero = ((i_feature * kernel) == 0)
                num_zero = torch.sum(is_zero, dim=1)
                num_zero_ops[:, j_cout, :, :] = num_zero.unsqueeze(0).reshape((bi, oh, ow))
            #print(num_zero_ops)
            #exit(1)
        else: # depthwise
            unfold = nn.Unfold(kernel_size=layer.kernel_size, padding=layer.padding, stride=layer.stride)
            for j_cout in range(cout): # output channel dimension
                image_channel = unfold(i_feature[:, j_cout].unsqueeze(1))
                kernel = weight[j_cout].flatten().unsqueeze(-1).unsqueeze(0).expand(bi, -1, oh*ow)
                is_zero = ((image_channel * kernel) == 0)
                num_zero = torch.sum(is_zero, dim=1)
                num_zero_ops[:, j_cout, :, :] = num_zero.unsqueeze(0).reshape((bi, oh, ow))
            #print(num_zero_ops)
            #exit(1)
        self.num_zero_ops[name] = torch.round(torch.mean(num_zero_ops, dim=0))
        #print(self.num_zero_ops[name])
        
    def count_zero_ops_linear(self, layer: nn.Conv2d, name:str):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        weight = layer.weight
        weight_quantized = self._get_quantized_weight(name)

        # size
        bi, cin  = i_feature.shape
        bo, cout = o_feature.shape
        assert bi == bo, 'ERROR! Batch size is not the same for input feature and output feature!'

        self.num_zero_input[name]  = round(torch.sum(i_feature == 0).item() / bi)
        self.num_zero_output[name] = round(torch.sum(o_feature == 0).item() / bi)
        self.num_zero_weight[name] = round(torch.sum(weight_quantized == 0).item())

        # count number of zero operations for every output pixel        
        i_patch = i_feature.unsqueeze(1).expand(-1, cout, -1)
        kernel  = weight.unsqueeze(0).expand(bi, -1, -1)
        is_zero = ((i_patch * kernel) == 0)
        num_zero_ops = torch.sum(is_zero, dim=-1).to(torch.float32)
        self.num_zero_ops[name] = torch.round(torch.mean(num_zero_ops, dim=0))
    
    def _get_quantized_weight(self, layer_name):
        for name, layer in self.model_q.named_modules():
            if ( layer_name == name ):
                w = layer.weight()
                wq = torch.int_repr(w)
                return wq
        raise Exception(f'ERROR! No quantized weight found for {layer_name}')

