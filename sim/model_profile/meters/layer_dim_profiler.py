"""
weight & input/output dimension profiler of Stripes
"""
import json


class LayerDim:
    def __init__(self, model_name: str) -> None:
        layer_dim_list = self._extract_layer_name(model_name)
        (self.layer_name_list, 
         self.weight_dim, self.input_dim, self.output_dim) = self._get_layer_info(layer_dim_list)
    
    def _extract_layer_name(self, model_name: str, group_size: int=32, col_num: int=4):
        base_path = '/home/yc2367/BitVert_DNN'
        model_config_path = f'{base_path}/BitVertZP_Grp{group_size}_N{col_num}/{model_name}'
        layer_dim_path = f'{model_config_path}/tensors/matmul.json'

        with open(layer_dim_path) as f:
            layer_dim_list = json.load(f)

        return layer_dim_list
    
    def _get_layer_info(self, layer_dim_list):
        layer_name_list = []
        weight_dim = {}
        input_dim  = {}
        output_dim = {}
        for name, layer_dim in layer_dim_list.items():
            layer_name = name.rstrip('.ops')
            if 'conv' in layer_name:
                layer_name_list.append(layer_name)

                bi, ci, hi, wi = layer_dim['x_shape']
                bo, co, ho, wo = layer_dim['z_shape']
                _, _, k, _ = layer_dim['y_shape']

                weight_dim[layer_name] = [k, k, ci, co]
                input_dim[layer_name]  = [bi, wi, hi, ci]
                output_dim[layer_name] = [bo, ho, wo, co]
            elif ('fc' in layer_name) or ('proj' in layer_name) or ('qkv' in layer_name):
                layer_name_list.append(layer_name)
                if len(layer_dim['x_shape']) == 2: # CNN
                    bi, ci = layer_dim['x_shape']
                    bo, co = layer_dim['z_shape']

                    weight_dim[layer_name] = [ci, co]
                    input_dim[layer_name] = [bi, 1, ci]
                    output_dim[layer_name] = [bo, 1, co]
                else:
                    bi, si, ci = layer_dim['x_shape']
                    bo, so, co = layer_dim['z_shape']

                    weight_dim[layer_name] = [ci, co]
                    input_dim[layer_name] = [bi, si, ci]
                    output_dim[layer_name] = [bo, so, co]
        return layer_name_list, weight_dim, input_dim, output_dim
