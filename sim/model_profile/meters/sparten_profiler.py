"""
weight & input/output zero operations profiler
"""
import torch
import torch.nn as nn
from torchvision import datasets
import pickle, os, math

from model_profile.meters.layer_dim_profiler import LayerDim


class SpartenProfiler(object):
    def __init__(self, model_name: str, group_size, args, device) -> None:
        self.model_name = model_name
        self.device = device
        self._init_model_profiler()
        (self.weight_tensor, 
         self.input_tensor, 
         self.output_tensor) = self._get_quantized_model() # quantized model

        self.save_dir = args.sparten_save_dir
        self.group_size = group_size
         
        self.num_zero_input  = {} 
        self.num_zero_output = {} 
        self.num_zero_weight = {} 
        # number of zero operations for every layer
        self.num_eff_ops = {}
        # check if a conv2d is followed by a ReLU activation function, 
        # this is used to measure output sparsity
        self.bn_list = {}
    
    def _init_model_profiler(self):
        dim_profiler = LayerDim(self.model_name)
        # format: {layer_name: [k, k, in_channel, out_channel], ...}
        self.weight_dim = dim_profiler.weight_dim 
        # format: {layer_name: [batch_size, width, height, in_channel], ...}
        self.input_dim  = dim_profiler.input_dim
        # format: {layer_name: [batch_size, width, height, out_channel], ...}
        self.output_dim = dim_profiler.output_dim
        self.layer_name_list = dim_profiler.layer_name_list
    
    def _get_quantized_model(self):
        weight_q = {}
        input_q  = {}
        output_q = {}
        base_path = '/home/yc2367/BitVert_DNN'
        model_config_path = f'{base_path}/Baseline_Int8/{self.model_name}'
        tensor_path = f'{model_config_path}/tensors'

        for name in self.layer_name_list:
            w_tensor_file = f'{tensor_path}/{name}.ops_x2.pt'
            w_tensor = torch.load(w_tensor_file).to(torch.float)
            if self.device == 'cuda':
                w_tensor = w_tensor.cuda()
            else:
                w_tensor = w_tensor.cpu()
            if len(w_tensor.shape) == 2: # transpose fully-connected layer
                w_tensor = w_tensor.permute(1, 0)
            weight_q[name] = w_tensor

            i_tensor_file = f'{tensor_path}/{name}.ops_x1.pt'
            i_tensor = torch.load(i_tensor_file).to(torch.float)
            if self.device == 'cuda':
                i_tensor = i_tensor.cuda()
            else:
                i_tensor = i_tensor.cpu()
            input_q[name] = i_tensor

            o_tensor_file = f'{tensor_path}/{name}.ops_y.pt'
            o_tensor = torch.load(o_tensor_file).to(torch.float)
            if self.device == 'cuda':
                o_tensor = o_tensor.cuda()
            else:
                o_tensor = o_tensor.cpu()
            output_q[name] = o_tensor
        return weight_q, input_q, output_q

    def fit(self, create_new_dict=False):
        save_dir = self.save_dir
        model_name = self.model_name
        file_num_zero_input  = save_dir + f'/{model_name}_num_zero_input.pickle'
        file_num_zero_output = save_dir + f'/{model_name}_num_zero_output.pickle'
        file_num_zero_weight = save_dir + f'/{model_name}_file_num_zero_weight.pickle'
        file_num_eff_ops     = save_dir + f'/{model_name}_file_num_eff_ops.pickle'
        dict_exists = os.path.isfile(file_num_zero_input) and \
                        os.path.isfile(file_num_zero_output) and \
                        os.path.isfile(file_num_zero_weight) and \
                        os.path.isfile(file_num_eff_ops)
        if create_new_dict or (not dict_exists):
            for name in self.layer_name_list:
                if len(self.weight_dim[name]) == 4:
                    self._calc_zero_value_conv(name)
                    self._calc_eff_ops_conv(name)
                else:
                    self._calc_zero_value_linear(name)
                    self._calc_eff_ops_linear(name)
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
    
    def _calc_zero_value_conv(self, name:str):
        i_feature = self.input_tensor[name]
        o_feature = self.output_tensor[name]
        weight_q = self.weight_tensor[name]
        bi, _, _, _ = i_feature.shape

        self.num_zero_weight[name] = round(torch.sum(weight_q == 0).item())
        self.num_zero_input[name]  = round(torch.sum(i_feature == 0).item() / bi)
        self.num_zero_output[name] = round(torch.sum(o_feature <= 0).item() / bi)
    
    def _calc_zero_value_linear(self, name:str):
        i_feature = self.input_tensor[name]
        o_feature = self.output_tensor[name]
        weight_q = self.weight_tensor[name]
        # size
        if len(i_feature.shape) == 2:
            bi, _  = i_feature.shape
        else:
            bi, _, _  = i_feature.shape
        self.num_zero_input[name]  = round(torch.sum(i_feature == 0).item() / bi)
        self.num_zero_output[name] = round(torch.sum(o_feature == 0).item() / bi)
        self.num_zero_weight[name] = round(torch.sum(weight_q == 0).item())

    def _calc_eff_ops_conv(self, name:str):
        i_feature = self.input_tensor[name]
        o_feature = self.output_tensor[name]
        weight_q = self.weight_tensor[name]
        group_size = self.group_size
        
        # size
        bi, _,  ih, iw = i_feature.shape
        bo, cout, oh, ow = o_feature.shape
        assert bi == bo, 'ERROR! Batch size is not the same for input feature and output feature!'
        # kernel
        _, cin, _, k = weight_q.shape
        # padding, stride
        stride = ih // oh
        padding = (oh * stride - stride + k - ih + 1) // 2

        # count number of zero operations for every group
        num_groups = math.ceil(k**2 * cin / group_size) 
        is_integer_num_group = ((k**2 * cin) % group_size) == 0
        if cin != 1: # conv2D
            num_eff_ops = torch.zeros([bo, cout, oh, ow, num_groups])
            unfold = nn.Unfold(kernel_size=k, padding=(padding, padding), stride=(stride, stride))
            i_feature = unfold(i_feature).permute([0, 2, 1])
            for j_cout in range(cout): # output channel dimension
                kernel = weight_q[j_cout].flatten()
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
            unfold = nn.Unfold(kernel_size=k, padding=padding, stride=stride)
            for j_cout in range(cout): # output channel dimension
                image_channel = unfold(i_feature[:, j_cout].unsqueeze(1))
                kernel = weight_q[j_cout].flatten().unsqueeze(-1).unsqueeze(0).expand(bi, -1, oh*ow)
                not_zero = ((image_channel * kernel) != 0)
                num_not_zero = torch.sum(not_zero, dim=1)
                num_eff_ops[:, j_cout, :, :] = num_not_zero.unsqueeze(-1).reshape((bi, oh, ow))
            num_eff_ops = num_eff_ops.unsqueeze(-1)
        self.num_eff_ops[name] = torch.round(torch.mean(num_eff_ops, dim=0))
        
    def _calc_eff_ops_linear(self, name:str):
        i_feature = self.input_tensor[name]
        o_feature = self.output_tensor[name]
        weight_q = self.weight_tensor[name]
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
            kernel = weight_q.unsqueeze(-1).reshape([cout, num_groups, group_size])
            kernel = kernel.unsqueeze(0).unsqueeze(0).expand(bi, si, -1, -1, -1)
            i_feature = i_feature.unsqueeze(-1).reshape([bi, si, num_groups, group_size])
            i_feature = i_feature.unsqueeze(2).expand([-1, -1, cout, -1, -1])
            not_zero = ((i_feature * kernel) != 0)
            num_eff_ops = torch.sum(not_zero, dim=-1).to(torch.float32)
        else:
            num_eff_ops = torch.zeros([bo, so, cout, num_groups])
            i_feature = i_feature.unsqueeze(2).expand(-1, -1, cout, -1)
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

