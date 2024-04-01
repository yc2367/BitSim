"""
weight & input/output zero operations profiler
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import pickle, os, math

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


class SpartenProfiler(Profiler):
    def __init__(self, name, model: nn.Module, group_size, args, device, precision=32) -> None:
        (self.testloader, 
         self.num_classes, 
         self.input_size) = get_loader(args)
        super().__init__(name, model, device, self.input_size, precision) 
        self.model_q = MODEL[name].cpu() # quantized model
        self.save_dir = args.sparten_save_dir
        self.group_size = group_size
         
        self.layer_name_list = []
        self.num_zero_input  = {} 
        self.num_zero_output = {} 
        self.num_zero_weight = {} 
        # number of zero operations for every layer
        self.num_eff_ops = {}
        # check if a conv2d is followed by a ReLU activation function, 
        # this is used to measure output sparsity
        self.bn_list = {}

    def hook(self):
        for n, m in self.model.named_modules():
            m.register_forward_hook(feature_hook(n, self.feature_dict))
    
    def forward(self):
        # hook
        self.hook()
        for (inputs, _) in self.testloader:
            inputs = inputs.cuda() if self.device == "cuda" else inputs
            # forward pass 
            with torch.no_grad():
                y = self.model(inputs)
            break

    def fit(self, create_new_dict=False):
        current_conv_layer = None
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.layer_name_list.append(n)
                current_conv_layer = n
            elif isinstance(m, nn.BatchNorm2d):
                self.bn_list[current_conv_layer] = m
            elif isinstance(m, nn.Linear):
                self.layer_name_list.append(n)

        save_dir = self.save_dir
        model_name = self.model_name
        file_num_zero_input  = save_dir + f'/{model_name}_num_zero_input.pickle'
        file_num_zero_output = save_dir + f'/{model_name}_num_zero_output.pickle'
        file_num_zero_weight = save_dir + f'/{model_name}_file_num_zero_weight.pickle'
        file_num_eff_ops    = save_dir + f'/{model_name}_file_num_eff_ops.pickle'
        dict_exists = os.path.isfile(file_num_zero_input) and \
                        os.path.isfile(file_num_zero_output) and \
                        os.path.isfile(file_num_zero_weight) and \
                        os.path.isfile(file_num_eff_ops)
        if create_new_dict or (not dict_exists):
            self.forward()
            for n, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d):
                    self._calc_zero_value_conv(m ,n)
                    self._calc_eff_ops_conv(m ,n)
                if isinstance(m, nn.Linear):
                    self._calc_zero_value_linear(m ,n)
                    self._calc_eff_ops_linear(m, n)
            with open(file_num_zero_input, 'wb') as f:
                pickle.dump(self.num_zero_input, f)
            with open(file_num_zero_output, 'wb') as f:
                pickle.dump(self.num_zero_output, f)
            with open(file_num_zero_weight, 'wb') as f:
                pickle.dump(self.num_zero_weight, f)
            with open(file_num_eff_ops, 'wb') as f:
                pickle.dump(self.num_eff_ops, f)
        else:
            with open(file_num_zero_input, 'rb') as f:
                self.num_zero_input = pickle.load(f)
            with open(file_num_zero_output, 'rb') as f:
                self.num_zero_output = pickle.load(f)
            with open(file_num_zero_weight, 'rb') as f:
                self.num_zero_weight = pickle.load(f)
            with open(file_num_eff_ops, 'rb') as f:
                self.num_eff_ops = pickle.load(f)
    
    def _calc_zero_value_conv(self, layer: nn.Conv2d, name:str):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        weight_quantized = self._get_quantized_weight(name)
        bi, _, _, _ = i_feature.shape

        self.num_zero_weight[name] = round(torch.sum(weight_quantized == 0).item())
        self.num_zero_input[name]  = round(torch.sum(i_feature == 0).item() / bi)
        if name in self.bn_list.keys():
            batchnorm = self.bn_list[name]
            o_feature = batchnorm(o_feature)
        self.num_zero_output[name] = round(torch.sum(o_feature <= 0).item() / bi)
    
    def _calc_zero_value_linear(self, layer: nn.Conv2d, name:str):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        weight_quantized = self._get_quantized_weight(name)

        # size
        if len(i_feature.shape) == 2:
            bi, _  = i_feature.shape
        else:
            bi, _, _  = i_feature.shape

        self.num_zero_input[name]  = round(torch.sum(i_feature == 0).item() / bi)
        self.num_zero_output[name] = round(torch.sum(o_feature == 0).item() / bi)
        self.num_zero_weight[name] = round(torch.sum(weight_quantized == 0).item())

    def _calc_eff_ops_conv(self, layer: nn.Conv2d, name:str):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        weight = layer.weight
        group_size = self.group_size
        
        # size
        bi, _,  _, _ = i_feature.shape
        bo, cout, oh, ow = o_feature.shape
        assert bi == bo, 'ERROR! Batch size is not the same for input feature and output feature!'
        # kernel
        k = layer.kernel_size[0]
        cin = layer.in_channels // layer.groups

        # count number of zero operations for every group
        num_groups = math.ceil(k**2 * cin / group_size) 
        is_integer_num_group = ((k**2 * cin) % group_size) == 0
        if cin == layer.in_channels: # conv2D
            num_eff_ops = torch.zeros([bo, cout, oh, ow, num_groups])
            unfold = nn.Unfold(kernel_size=layer.kernel_size, padding=layer.padding, stride=layer.stride)
            i_feature = unfold(i_feature).permute([0, 2, 1])
            for j_cout in range(cout): # output channel dimension
                kernel = weight[j_cout].flatten()
                if is_integer_num_group:
                    kernel = kernel.unsqueeze(-1).reshape([num_groups, group_size])
                    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(bi, oh*ow, -1, -1)
                    i_feature = i_feature.unsqueeze(-1).reshape([bi, oh*ow, num_groups, group_size])
                    not_zero = ((i_feature * kernel) != 0)
                    num_not_zero = torch.sum(not_zero, dim=-1)
                    num_eff_ops[:, j_cout, :, :, :] = num_not_zero.reshape((bi, oh, ow, num_groups))
                else:
                    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(bi, oh*ow, -1)
                    for j_group in range(num_groups):
                        # divide the dot-product into groups of group_size
                        l_j_group = j_group * group_size
                        u_j_group = (j_group+1) * group_size
                        i_patch = i_feature[:, :, l_j_group:u_j_group]
                        w_patch = kernel[:, :, l_j_group:u_j_group]
                        not_zero = ((i_patch * w_patch) != 0)
                        num_not_zero = torch.sum(not_zero, dim=-1)
                        num_eff_ops[:, j_cout, :, :, j_group] = num_not_zero.unsqueeze(1).reshape((bi, oh, ow))
        else: # depthwise
            num_eff_ops = torch.zeros([bo, cout, oh, ow])
            unfold = nn.Unfold(kernel_size=layer.kernel_size, padding=layer.padding, stride=layer.stride)
            for j_cout in range(cout): # output channel dimension
                image_channel = unfold(i_feature[:, j_cout].unsqueeze(1))
                kernel = weight[j_cout].flatten().unsqueeze(-1).unsqueeze(0).expand(bi, -1, oh*ow)
                not_zero = ((image_channel * kernel) != 0)
                num_not_zero = torch.sum(not_zero, dim=1)
                num_eff_ops[:, j_cout, :, :] = num_not_zero.unsqueeze(-1).reshape((bi, oh, ow))
            num_eff_ops = num_eff_ops.unsqueeze(-1)
        self.num_eff_ops[name] = torch.round(torch.mean(num_eff_ops, dim=0))
        
    def _calc_eff_ops_linear(self, layer: nn.Linear, name:str):
        i_feature = self.feature_dict[name][0]
        o_feature = self.feature_dict[name][1]
        weight = layer.weight
        group_size = self.group_size

        # size
        if len(i_feature.shape) == 2:
            bi, cin  = i_feature.shape
            bo, cout = o_feature.shape
            si = 1
            so = 1
            i_feature = i_feature.unsqueeze(1) # dim: [batch_size, sample_size, input feature]
        elif len(i_feature.shape) == 3:
            bi, si, cin = i_feature.shape
            bo, so, cout = o_feature.shape
        else:
            raise Exception('ERROR! More than 3 dimensions is provided for linear layer!')
        
        assert bi == bo, 'ERROR! Batch size is not the same for input feature and output feature!'

        num_groups = math.ceil(cin / group_size) 
        is_integer_num_group = (cin % group_size) == 0

        # count number of zero operations for every group
        if is_integer_num_group:
            kernel = weight.unsqueeze(-1).reshape([cout, num_groups, group_size])
            kernel = kernel.unsqueeze(0).unsqueeze(0).expand(bi, si, -1, -1, -1)
            i_feature = i_feature.unsqueeze(-1).reshape([bi, si, num_groups, group_size])
            i_feature = i_feature.unsqueeze(1).expand([-1, -1, cout, -1, -1])
            not_zero = ((i_feature * kernel) != 0)
            num_eff_ops = torch.sum(not_zero, dim=-1).to(torch.float32)
        else:
            num_eff_ops = torch.zeros([bo, so, cout, num_groups])
            i_feature = i_feature.unsqueeze(1).expand(-1, -1, cout, -1)
            kernel  = weight.unsqueeze(0).unsqueeze(0).expand(bi, si, -1, -1)
            for j_group in range(num_groups):
                # divide the dot-product into groups of group_size
                l_j_group = j_group * num_groups
                u_j_group = (j_group+1) * num_groups
                i_patch = i_feature[:, :, :, l_j_group:u_j_group]
                w_patch = kernel[:, :, :, l_j_group:u_j_group]
                not_zero = ((i_patch * w_patch) != 0)
                num_eff_ops[:, :, :, j_group] = torch.sum(not_zero, dim=-1).to(torch.float32)
        self.num_eff_ops[name] = torch.round(torch.mean(num_eff_ops, dim=0))

    def _get_quantized_weight(self, layer_name):
        for name, layer in self.model_q.named_modules():
            if ( layer_name == name ):
                w = layer.weight()
                wq = torch.int_repr(w)
                return wq
        raise Exception(f'ERROR! No quantized weight found for {layer_name}')

