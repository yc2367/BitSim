"""
non-zero channels profiler of BitVert
"""
import json


class BitVertProfiler(object):
    def __init__(self, model_name) -> None:
        self.nonzero_channels = self._extract_nonzero_channels(model_name)
    
    def _extract_nonzero_channels(self, model_name: str, group_size: int=32, col_num: int=4):
        base_path = '/home/yc2367/BitVert_DNN'
        model_config_path = f'{base_path}/BitVertZP_Grp{group_size}_N{col_num}/{model_name}'
        nonzero_channels_file = f'{model_config_path}/nonzero_channels.json'

        with open(nonzero_channels_file) as f:
            nonzero_channels = json.load(f)

        return nonzero_channels
