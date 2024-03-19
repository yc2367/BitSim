from typing import List

class ArrayDimension:
    ## The class constructor
    ## @param index: The integer index of this Dimension.
    ## @param name: The user-provided name of this Dimension.
    ## @param size: The user-provided size of this Dimension.
    def __init__(self, index: int, name: str, size: int):
        self.id = index
        self.name = name
        self.size = size

    def __str__(self):
        return f"Dimension (id={self.id}, name={self.name}, size={self.size})"

    def __repr__(self):
        return str(self)

    ## JSON representation of this class to save it to a json file.
    def __jsonrepr__(self):
        return self.__dict__

    def __eq__(self, __o: object):
        if not isinstance(__o, ArrayDimension):
            return False
        return (
            ( self.id == __o.id ) and 
            ( self.name == __o.name ) and 
            ( self.size == __o.size )
        )

    def __hash__(self):
        return hash(self.id) ^ hash(self.name)

