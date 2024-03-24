from typing import List
import math

class OperationalUnit:
    ## The class constructor
    # @param input_precision:  The bit precision of the operation inputs.
    # @param output_precision: The bit precision of the operation outputs.
    # @param unit_energy:      The energy cost of performing a single operation.
    # @param unit_area:        The area of a single operational unit.
    # @param include_energy:   If True, then use unit_energy for the energy. 
    #                          If False, then the unit_energy is included in the PE energy.
    # @param include_area:     If True, then use unit_area for the area. 
    #                          If False, then the unit_area is included in the PE area.
    def __init__(
        self,
        input_precision: List[int],
        output_precision: int,
        unit_energy: float,
        unit_area: float,
        include_energy: bool,
        include_area: bool
    ):
        if len(input_precision) == 1:
            self.input_precision  = input_precision[0]
        else:
            self.input_precision  = input_precision
        self.output_precision = output_precision
        self.precision        = input_precision + [output_precision]

        if include_energy:
            self.energy = unit_energy
        else:
            self.energy = 0

        if include_area:
            self.area = unit_area
        else:
            self.area = 0

    ## JSON Representation of this class to save it to a json file.
    def __jsonrepr__(self):
        return self.__dict__

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, OperationalUnit):
            return False
        return (
            ( self.precision      == __o.precision )     and 
            ( self.energy         == __o.energy )        and
            ( self.area           == __o.area )            
        )
    
    def get_precision(self):
        return self.precision
    
    def get_area(self):
        return self.area
    
    def get_energy(self):
        return self.energy


## Hardware multiplier
class Multiplier(OperationalUnit):
    ## The class constructor
    # @param input_precision: The bit precision of the multiplication inputs.
    # @param energy_cost:     The energy cost of performing a single multiplication.
    # @param area:            The area of a single multiplier.
    # @param include_energy:  If True, then use energy_cost for the multiplier energy. 
    #                         If False, then the energy_cost is included in the PE energy.
    # @param include_area:    If True, then use area for the multiplier area. 
    #                         If False, then the area is included in the PE area.
    def __init__(
            self, 
            input_precision: List[int], 
            energy_cost: float, 
            area: float,
            include_energy: bool,
            include_area: bool
    ):
        if len(input_precision) > 2:
            print(f'ERROR! The length of the input_precision list is {len(input_precision)}, which is more than 2.')

        if 1 in input_precision:
            output_precision = max(input_precision)
        else:
            output_precision = sum(input_precision)
            
        super().__init__(input_precision, output_precision, energy_cost, area, include_energy, include_area)


## Hardware adder
class Adder(OperationalUnit):
    ## The class constructor
    # @param input_precision: The bit precision of the addition inputs.
    # @param energy_cost:     The energy cost of performing a single addition.
    # @param area:            The area of a single adder.
    # @param include_energy:  If True, then use energy_cost for the adder energy. 
    #                         If False, then the energy_cost is included in the PE energy.
    # @param include_area:    If True, then use area for the adder area. 
    #                         If False, then the area is included in the PE area.
    def __init__(
            self, 
            input_precision: List[int], 
            energy_cost: float, 
            area: float,
            include_energy: bool,
            include_area: bool
    ):
        if len(input_precision) > 2:
            print(f'ERROR! The length of the input_precision list is {len(input_precision)}, which is more than 2.')
        output_precision = max(input_precision) + 1
        super().__init__(input_precision, output_precision, energy_cost, area, include_energy, include_area)
    

## Hardware register
class Register(OperationalUnit):
    ## The class constructor
    # @param input_precision: The bit precision of the register inputs.
    # @param energy_cost:     The energy cost of register.
    # @param area:            The area of a single register.
    # @param include_energy:  If True, then use energy_cost for the register energy. 
    #                         If False, then the energy_cost is included in the PE energy.
    # @param include_area:    If True, then use area for the register area. 
    #                         If False, then the area is included in the PE area.
    def __init__(
            self, 
            input_precision: List[int], 
            energy_cost: float, 
            area: float,
            include_energy: bool,
            include_area: bool
    ):
        if len(input_precision) != 1:
            print(f'ERROR! The length of the input_precision list can only be 1 for a register, not {len(input_precision)}.')
        output_precision = input_precision
        super().__init__(input_precision, output_precision, energy_cost, area, include_energy, include_area)


## Bit Serial PE
class PE(OperationalUnit):
    ## The class constructor
    # @param input_precision: The bit precision of the bit-serial PE.
    # @param group_size:      The group size of the bit-serial PE.
    # @param energy_cost:     The energy cost of PE.
    # @param area:            The area of PE.
    def __init__(
            self, 
            input_precision: List[int], 
            group_size: int,
            energy_cost: float, 
            area: float
    ):
        if ( len(input_precision) != 2 ):
            print(f'ERROR! You must provide precision for 2 input operands of a bit-serial PE.')
            exit(1)
        if ( energy_cost == 0 ):
            print(f'ERROR! You must provide the energy cost of a PE.')
            exit(1)
        if ( area == 0 ):
            print(f'ERROR! You must provide the area of a PE.')
            exit(1)

        self.group_size = group_size
        include_energy = True
        include_area   = True
        output_precision = input_precision[0] + input_precision[1] + math.log2(group_size)
        super().__init__(input_precision, output_precision, 
                         energy_cost, area, include_energy, include_area)


## Bit Serial PE
class BitSerialPE(PE):
    ## The class constructor
    # @param input_precision_s: The precision of bit-serial input.
    # @param input_precision_p: The precision of bit-parallel input.
    # @param group_size:        The group size of the bit-serial PE.
    # @param energy_cost:       The energy cost of the bit-serial PE.
    # @param area:              The area of the bit-serial PE.
    def __init__(
            self, 
            input_precision_s: int, 
            input_precision_p: int, 
            group_size: int,
            energy_cost: float, 
            area: float
    ):
        if ( energy_cost == 0 ):
            print(f'ERROR! You must provide the energy cost of a PE.')
            exit(1)
        if ( area == 0 ):
            print(f'ERROR! You must provide the area of a PE.')
            exit(1)
        input_precision = [input_precision_s, input_precision_p]
        self.input_precision_s = input_precision_s
        self.input_precision_p = input_precision_p
        super().__init__(input_precision, group_size, energy_cost, area)
