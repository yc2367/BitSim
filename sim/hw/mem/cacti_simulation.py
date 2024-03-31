import yaml
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

class CactiSimulation:

    ## Path of current directory
    cacti_top_path = os.path.dirname(os.path.realpath(__file__))
    ## Path to cached cacti simulated memories
    MEM_POOL_PATH = f"{cacti_top_path}/example_cacti_pool.yaml"
    ## Path to cacti python script to extract costs
    CACTI_TOP_PATH = f"{cacti_top_path}/cacti_program.py"

    ## The class constructor
    def __init__(self):
        if os.path.isfile(self.MEM_POOL_PATH):
            os.remove(self.MEM_POOL_PATH)
        open(self.MEM_POOL_PATH, 'w').close()
        self.required_keys = ['technology', 'mem_type', 'size', 'bank_count', 'rw_bw', 'r_port', 'w_port', 'rw_port']
    
    def is_valid_config(self, mem_config):
        provided_keys = mem_config.keys()
        missed_keys = []
        for key in self.required_keys:
            if key not in provided_keys:
                missed_keys.append(key)
        if len(missed_keys) != 0:
            return False, missed_keys
        return True, missed_keys
    
    ## This function checks if the provided memory configuration was already used in the past.
    # @param mem_config: configuration of memory
    # @param mem_pool_path  Path to cached cacti simulated memories
    # @return True          The requested memory item has been simulated once.
    # @return False         The requested memory item has not been simualted so far.
    def item_exists(
        self,
        mem_config,
        mem_pool_path=MEM_POOL_PATH,
    ):
        is_valid, missed_keys = self.is_valid_config(mem_config)
        if not is_valid:
            raise ValueError(f'The provided memory configuration is INVALID. Missed keys: {str(missed_keys)}')
        
        with open(mem_pool_path, "r") as fp:
            memory_pool = yaml.full_load(fp)

        if memory_pool != None:
            for instance in memory_pool:
                bank_count = int(mem_config["bank_count"])
                technology = memory_pool[instance]["technology"]
                mem_type = memory_pool[instance]["memory_type"]
                cache_size = int(memory_pool[instance]["size_bit"] * bank_count)
                IO_bus_width = int(memory_pool[instance]["IO_bus_width"] * bank_count) 
                ex_rd_port = int(memory_pool[instance]["ex_rd_port"])
                ex_wr_port = int(memory_pool[instance]["ex_wr_port"])
                rd_wr_port = int(memory_pool[instance]["rd_wr_port"])

                if (
                    (mem_config['mem_type'] == mem_type)
                    and (mem_config['size'] == cache_size)
                    and (mem_config['bank_count'] == bank_count)
                    and (mem_config['rw_bw'] == IO_bus_width)
                    and (mem_config['r_port'] == ex_rd_port)
                    and (mem_config['w_port'] == ex_wr_port)
                    and (mem_config['rw_port'] == rd_wr_port)
                    and (mem_config['technology'] == technology)
                ):
                    return True

        return False

    ## This function simulates a new item by calling CACTI based on the provided parameters
    # @param mem_config: configuration of memory
    # @param mem_pool_path  Path to cached cacti simulated memories
    # @param cacti_top_path Path to cacti python script to extract costs
    def create_item(
        self,
        mem_config,
        mem_pool_path=MEM_POOL_PATH,
        cacti_top_path=CACTI_TOP_PATH,
    ):
        is_valid, missed_keys = self.is_valid_config(mem_config)
        if not is_valid:
            raise ValueError(f'The provided memory configuration is INVALID. Missed keys: {str(missed_keys)}')
        
        array_size = int(mem_config['size'] / 8 / mem_config['bank_count'])
        array_IO_bus_width = int(mem_config['rw_bw'] / mem_config['bank_count'])        
        try:
            output = subprocess.check_output(
                [
                    "python", cacti_top_path,
                    "--technology", str(mem_config['technology']),
                    "--mem_type", mem_config['mem_type'],
                    "--cache_size", str(array_size),
                    "--bank_count", str(1),
                    "--IO_bus_width", str(array_IO_bus_width),
                    "--ex_rd_port", str(mem_config['r_port']),
                    "--ex_wr_port", str(mem_config['w_port']),
                    "--rd_wr_port", str(mem_config['rw_port']),
                    "--mem_pool_path", str(mem_pool_path),
                ],
                stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as exc:
            print(f"Cacti subprocess call failed.")
            f = exc.output.decode('utf-8')
            print(f)

    ## This functions checks first if the memory with the provided parameters was already simulated once.
    # In case it hasn't been simulated, then it will create a new memory item based on the provided parameters.
    # @param mem_config: configuration of memory
    # @param mem_pool_path  Path to cached cacti simulated memories
    # @param cacti_top_path Path to cacti python script to extract costs
    def get_item(
        self,
        mem_config,
        mem_pool_path=MEM_POOL_PATH,
        cacti_top_path=CACTI_TOP_PATH,
    ):
        is_valid, missed_keys = self.is_valid_config(mem_config)
        if not is_valid:
            raise ValueError(f'The provided memory configuration is INVALID. Missed keys: {str(missed_keys)}')
        
        if not os.path.exists(cacti_top_path):
            raise FileNotFoundError(f"Cacti top file doesn't exist: {cacti_top_path}.")

        logger.info(
            f"Extracting memory costs with CACTI for size = {mem_config['size']} and rw_bw = {mem_config['rw_bw']}."
        )

        if mem_config['mem_type'] == "rf":
            new_mem_type = "sram"
            new_size = int(mem_config['size'] * 128)
            
            logger.warning(
                f"Type {mem_config['mem_type']} -> {new_mem_type}. Size {mem_config['size']} -> {new_size}." 
            )

            mem_config['mem_type'] = new_mem_type
            mem_config['size'] = new_size
        
        self.create_item(mem_config, mem_pool_path, cacti_top_path)

        with open(mem_pool_path, "r") as fp:
            memory_pool = yaml.full_load(fp)

        if memory_pool != None:
            for instance in memory_pool:
                bank_count = int(mem_config["bank_count"])
                technology = memory_pool[instance]["technology"]
                mem_type = memory_pool[instance]["memory_type"]
                cache_size = int(memory_pool[instance]["size_bit"] * bank_count)
                IO_bus_width = int(memory_pool[instance]["IO_bus_width"] * bank_count) 
                ex_rd_port = int(memory_pool[instance]["ex_rd_port"])
                ex_wr_port = int(memory_pool[instance]["ex_wr_port"])
                rd_wr_port = int(memory_pool[instance]["rd_wr_port"])
                read_cost = memory_pool[instance]["cost"]["read_word"] * bank_count * 1000
                write_cost = memory_pool[instance]["cost"]["write_word"] * bank_count * 1000
                area = memory_pool[instance]["area"] * bank_count
                latency = memory_pool[instance]["latency"]

                if (
                    (mem_config['mem_type'] == mem_type)
                    and (mem_config['size'] == cache_size)
                    and (mem_config['bank_count'] == bank_count)
                    and (mem_config['rw_bw'] == IO_bus_width)
                    and (mem_config['r_port'] == ex_rd_port)
                    and (mem_config['w_port'] == ex_wr_port)
                    and (mem_config['rw_port'] == rd_wr_port)
                    and (mem_config['technology'] == technology)
                ):
                    mem_config['r_cost'] = read_cost
                    mem_config['w_cost'] = write_cost
                    mem_config['area'] = area
                    mem_config['latency'] = latency

                    return mem_config

        # should be never reached
        raise ModuleNotFoundError(
            f"No match in Cacti memory pool found " + 
            f"technology = {mem_config['technology']}, " + 
            f"size = {mem_config['size']}, " + 
            f"bank = {mem_config['bank_count']}, " + 
            f"rw_bw = {mem_config['rw_bw']}, " + 
            f"r_port = {mem_config['r_port']}, " + 
            f"w_port = {mem_config['w_port']}, " + 
            f"rw_port = {mem_config['rw_port']}"
        )