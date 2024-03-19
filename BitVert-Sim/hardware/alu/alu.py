from typing import List

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
            ( self.area           == __o.area )          and 
            ( self.include_energy == __o.include_energy) and 
            ( self.include_area   == __o.include_area)    
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


