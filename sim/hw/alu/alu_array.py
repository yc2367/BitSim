from typing import List, Dict
import numpy as np
from hw.alu.alu_util import ArrayDimension
from hw.alu.alu_unit import (
    OperationalUnit,
    Multiplier,
    Adder,
    PE,
    BitSerialPE
)


class OperationalArray:
    ## The class constructor
    # @param pe:        The operational unit of the array.
    # @param dimension: The dimention of the array.
    def __init__(
        self,
        pe:        OperationalUnit,
        dimension: Dict[str, int]
    ):
        self.pe = pe
        self.total_unit_count = int(np.prod(list(dimension.values())))
        self.dimension = [
            ArrayDimension(idx, name, size)
            for idx, (name, size) in enumerate(dimension.items())
        ]
        self.dimension_size = [dim.size for dim in self.dimension]
        self.num_dimension = len(self.dimension)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, OperationalArray):
            return False
        return ( self.pe == __o.pe ) and ( self.dimension == __o.dimension )
    
    def get_precision(self):
        return self.pe.get_precision()
    
    def get_area(self):
        return self.total_unit_count * self.pe.get_area()
    
    def get_energy(self):
        return self.total_unit_count * self.pe.get_energy()


class AdderTreeLevel(OperationalArray):
    ## The class constructor
    # @param level:  The level of the adder tree.
    # @param fan_in: The fan_in of the this level.
    # @param adder:  The adder unit.
    def __init__(
        self,
        adder: Adder,
        level: int,
        fan_in: int
    ):
        dimension = {str(level): np.ceil(fan_in/2)}
        super().__init__(adder, dimension)


class AdderTree:
    ## The class constructor
    # @param level:  The level of the adder tree.
    # @param fan_in: The fan_in of the this level.
    # @param adder:  The adder unit.
    def __init__(
        self,
        fan_in: int,
        input_precision: List[int],
        unit_energy: float,
        unit_area: float,
        include_energy: bool,
        include_area: bool
    ):
        if len(input_precision) != 1:
            print(f'ERROR! The length of the input_precision list can only be 1 for an adder tree, not {len(input_precision)}.')
        self.fan_in = fan_in
        self.input_precision = input_precision
        self.pe_energy = unit_energy
        self.pe_area = unit_area
        self.levels = self._create_adder_tree(include_energy, include_area)
        self.num_levels = len(self.levels)
        self.output_precision = self.levels[-1].pe.output_precision

    def _create_adder_tree(self, include_energy, include_area):
        adder_levels = []
        fan_in = self.fan_in
        num_level = int(np.floor(np.log2(fan_in)))
        for level in range(num_level):
            input_precision = self.input_precision[0] + level
            scaling_ratio = input_precision / self.input_precision[0]
            area = self.pe_area * scaling_ratio
            energy = self.pe_energy * scaling_ratio
            adder = Adder([input_precision], energy, area, include_energy, include_area)
            adderTree = AdderTreeLevel(adder, level, fan_in)
            adder_levels.append(adderTree)
            fan_in = fan_in // 2
        
        return adder_levels
    
    def get_energy(self):
        return sum([level.get_energy() for level in self.levels])
    
    def get_area(self):
        return sum([level.get_area() for level in self.levels])


class PEArray(OperationalArray):
    ## The class constructor
    # @param dimension: The dimension of the PE array.
    # @param pe:        The processing element.
    def __init__(
        self,
        pe: PE,
        dimension: Dict[str, int]
    ):
        super().__init__(pe, dimension)


class BitSerialPEArray(PEArray):
    ## The class constructor
    # @param dimension:  The dimension of the multiplier array.
    # @param pe:         The bit-serial pe.
    def __init__(
        self,
        pe: BitSerialPE,
        dimension: Dict[str, int]
    ):
        super().__init__(pe, dimension)


if __name__ == "__main__":
    array = AdderTree(64, [8], 10, 20, True, True)
    print(f'input precision {array.input_precision}')
    print(f'output precision: {array.output_precision}')
    print(f'number of adder tree levels: {array.num_levels}')
    print(f'area: {array.get_energy()}')
    print(f'energy: {array.get_area()}')
    for i in range(array.num_levels):
        adder_level = array.levels[i]
        print(adder_level.total_unit_count, adder_level.pe.get_energy(), 
              adder_level.pe.get_area(), adder_level.get_energy(), 
              adder_level.get_area(), adder_level.pe.get_precision())
    
    pe = BitSerialPE(5, 8, 16, 3, 1)
    array = BitSerialPEArray(pe, {'h': 32, 'w': 32})
    print(array.dimension, array.get_energy())
