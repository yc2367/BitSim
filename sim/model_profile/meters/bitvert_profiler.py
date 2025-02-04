"""
non-zero channels profiler of BitVert
"""
import json


class BitVertProfiler(object):
    def __init__(self, model_name, en_eager_comp: bool=True) -> None:
        self.nonzero_channels = self._extract_nonzero_channels(model_name, en_eager_comp)
    
    def _extract_nonzero_channels(self, model_name: str, en_eager_compression, group_size: int=32):
        base_path = '/home/yc2367/BitVert_DNN'
        if en_eager_compression:
            col_num = 4
            model_config_path = f'{base_path}/BitVertZP_Grp{group_size}_N{col_num}/{model_name}'
        else:
            col_num = 2
            model_config_path = f'{base_path}/BitVertCA_Grp{group_size}_N{col_num}/{model_name}'
        nonzero_channels_file = f'{model_config_path}/nonzero_channels.json'

        with open(nonzero_channels_file) as f:
            nonzero_channels = json.load(f)

        return nonzero_channels
