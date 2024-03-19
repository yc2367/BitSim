from cacti_parser import CactiParser
from typing import Dict

## Description missing
class MemoryInstance:
    ## The class constructor
    # Collect all the basic information of a physical memory module.
    # @param name: memory module name, e.g. 'SRAM_512KB_BW_16b', 'I_RF'.
    # @param size: total memory capacity (unit: bit).
    # @param rw_bw/w_bw: memory bandwidth (or wordlength) (unit: bit/cycle).
    # @param r_cost/w_cost: memory unit data access energy.
    # @param area: memory area (unit can be whatever user-defined unit).
    # @param r_port: number of memory read port.
    # @param w_port: number of memory write port (rd_port and wr_port can work in parallel).
    # @param rw_port: number of memory port for both read and write (read and write cannot happen in parallel).
    # @param latency: memory access latency (unit: number of cycles).
    # @param min_r_granularity (int): The minimal number of bits than can be read in a clock cycle (can be a less than rw_bw)
    # @param min_w_granularity (int): The minimal number of bits that can be written in a clock cycle (can be less than w_bw)
    # @param mem_type (str): The type of memory. Used for CACTI cost extraction.
    # @param technology (float): The technology node.
    # @param auto_cost_extraction (bool): Automatically extract the read cost, write cost and area using CACTI.
    # @param double_buffering_support (bool): Support for double buffering on this memory instance.
    def __init__(
        self,
        name: str,
        mem_config: Dict,
        r_cost: float = 0,
        w_cost: float = 0,
        latency: int = 1,
        area: float = 0,
        min_r_granularity=None,
        min_w_granularity=None,
        auto_cost_extraction: bool = False,
        double_buffering_support: bool = False,
    ):
        if auto_cost_extraction:
            # Size must be a multiple of 8 when using CACTI
            assert (
                mem_config['size'] % 8 == 0
            ), "Memory size must be a multiple of 8 when automatically extracting costs using CACTI."
            cacti_parser = CactiParser()
            
            mem_config = cacti_parser.get_item(mem_config)
            self.r_cost = round(mem_config['r_cost'], 2)
            self.w_cost = round(mem_config['w_cost'], 2)
            self.area = mem_config['area']
        else:
            self.r_cost = r_cost
            self.w_cost = w_cost
            self.area = area
        
        self.size = mem_config['size']
        self.bank = mem_config['bank_count']
        self.rw_bw = mem_config['rw_bw']
        self.r_port = mem_config['r_port']
        self.w_port = mem_config['w_port']
        self.rw_port = mem_config['rw_port']

        self.name = name
        self.latency = latency
        self.double_buffering_support = double_buffering_support
        if not min_r_granularity:
            self.rw_bw_min = mem_config['rw_bw']
        else:
            self.rw_bw_min = min_r_granularity
        if not min_w_granularity:
            self.w_bw_min = mem_config['rw_bw']
        else:
            self.w_bw_min = min_w_granularity

    ## JSON Representation of this class to save it to a json file.
    def __jsonrepr__(self):
        return self.__dict__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MemoryInstance) and self.__dict__ == other.__dict__

    def __hash__(self):
        return id(self)  # unique for every object within its lifetime

    def __str__(self):
        return f"MemoryInstance({self.name})"

    def __repr__(self):
        return str(self)


if __name__ == "__main__":
    mem_config = {'technology': 0.028,
                  'mem_type': 'sram', 
                  'size': 131072 * 8, 
                  'bank_count': 1, 
                  'rw_bw': 128, 
                  'r_port': 1, 
                  'w_port': 1, 
                  'rw_port': 0
                  }
    mem = MemoryInstance('test_mem', mem_config, 0, 0, 1, 0, None, None, True, False)
    print(f'read cost: {mem.r_cost}, write cost: {mem.w_cost}, area: {mem.area}')
    '''
    name: str,
        mem_config,
        r_cost: float = 0,
        w_cost: float = 0,
        latency: int = 1,
        area: float = 0,
        min_r_granularity=None,
        min_w_granularity=None,
        mem_type: str = "sram",
        auto_cost_extraction: bool = False,
        double_buffering_support: bool = False,
    '''