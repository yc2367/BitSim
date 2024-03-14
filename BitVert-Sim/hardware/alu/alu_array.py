from typing import List, Dict
import numpy as np
from alu_util import ArrayDimension
from alu import (
    OperationalUnit,
    Multiplier,
    Adder
)


class OperationalArray:
    ## The class constructor
    # @param unit:      The operational unit of the array.
    # @param dimension: The dimention of the array.
    def __init__(
        self,
        dimension:        Dict[str, int],
        operational_unit: OperationalUnit
    ):
        self.unit = operational_unit
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
        return ( self.unit == __o.unit ) and ( self.dimension == __o.dimension )
    
    def get_precision(self):
        return self.unit.get_precision()
    
    def get_area(self):
        return self.total_unit_count * self.unit.get_area()
    
    def get_energy(self):
        return self.total_unit_count * self.unit.get_energy()
        

class MultiplierArray(OperationalArray):
    ## The class constructor
    # @param dimension:   The dimension of the multiplier array.
    # @param multiplier:  The multiplier unit.
    def __init__(
        self,
        dimension: Dict[str, int],
        multiplier: Multiplier
    ):
        super().__init__(dimension, multiplier)


class AdderTreeLevel(OperationalArray):
    ## The class constructor
    # @param level:  The level of the adder tree.
    # @param fan_in: The fan_in of the this level.
    # @param adder:  The adder unit.
    def __init__(
        self,
        level: int,
        fan_in: int,
        adder: Adder
    ):
        dimension = {str(level): np.ceil(fan_in/2)}
        super().__init__(dimension, adder)


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
        self.unit_energy = unit_energy
        self.unit_area = unit_area
        self.levels = self._create_adder_tree(include_energy, include_area)
        self.num_levels = len(self.levels)
        self.output_precision = self.levels[-1].unit.output_precision

    def _create_adder_tree(self, include_energy, include_area):
        adder_levels = []
        fan_in = self.fan_in
        num_level = int(np.floor(np.log2(fan_in)))
        for level in range(num_level):
            input_precision = self.input_precision[0] + level
            scaling_ratio = input_precision / self.input_precision[0]
            area = self.unit_area * scaling_ratio
            energy = self.unit_energy * scaling_ratio
            adder = Adder([input_precision], energy, area, include_energy, include_area)
            adderTree = AdderTreeLevel(level, fan_in, adder)
            adder_levels.append(adderTree)
            fan_in = fan_in // 2
        
        return adder_levels
    
    def get_energy(self):
        return sum([level.get_energy() for level in self.levels])
    
    def get_area(self):
        return sum([level.get_area() for level in self.levels])


if __name__ == "__main__":
    array = AdderTree(64, [8], 10, 20, True, True)
    print(f'input precision {array.input_precision}')
    print(f'output precision: {array.output_precision}')
    print(f'number of adder tree levels: {array.num_levels}')
    print(f'area: {array.get_energy()}')
    print(f'energy: {array.get_area()}')
    for i in range(array.num_levels):
        adder_level = array.levels[i]
        print(adder_level.total_unit_count, adder_level.unit.get_energy(), adder_level.unit.get_area(), adder_level.get_energy(), adder_level.get_area(), adder_level.unit.get_precision())
