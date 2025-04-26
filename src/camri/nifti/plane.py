import numpy as np
from enum import Enum
from typing import Union, Tuple

class Plane(Enum):
    SAGITTAL = 0
    CORONAL = 1
    AXIAL = 2

    @classmethod
    def _parse(cls, plane: Union[str, int, 'Plane']) -> 'Plane':
        """
        Internal helper method: Converts a string, integer, or Plane instance into a Plane enum.
        """
        if isinstance(plane, cls):
            return plane
        elif isinstance(plane, str):
            key = plane.strip().lower()
            mapping = {
                'sagittal': cls.SAGITTAL,
                'coronal': cls.CORONAL,
                'axial': cls.AXIAL
            }
            try:
                return mapping[key]
            except KeyError:
                raise ValueError("Plane must be 'axial', 'coronal', or 'sagittal'.")
        elif isinstance(plane, int):
            try:
                return cls(plane)
            except ValueError:
                raise ValueError("Plane must be an integer between 0 and 2.")
        else:
            raise ValueError("Plane must be a string, integer, or Plane enum.")

    @classmethod
    def from_string(cls, s: str) -> 'Plane':
        """
        Converts a string into a Plane enum.
        """
        return cls._parse(s)

    @classmethod
    def from_value(cls, v: int) -> 'Plane':
        """
        Converts an integer into a Plane enum.
        """
        return cls._parse(v)

    @classmethod
    def to_string(cls, v: Union[int, 'Plane']) -> str:
        """
        Converts an integer or Plane enum into a string.
        """
        plane_enum = cls._parse(v)
        mapping = {
            cls.SAGITTAL: 'sagittal',
            cls.CORONAL: 'coronal',
            cls.AXIAL: 'axial'
        }
        return mapping[plane_enum]

    @classmethod
    def get_indexer(cls, plane: Union[str, int, 'Plane'], idx: int) -> Tuple:
        """
        Assigns idx to the axis corresponding to the given plane,
        and assigns slice(None) to the remaining axes, returning an index tuple.
        For example, if the plane is 'axial' (2), returns (slice(None), slice(None), idx).
        """
        plane_enum = cls._parse(plane)
        return tuple(idx if i == plane_enum.value else slice(None) for i in range(3))

    @classmethod
    def get_others(cls, plane: Union[str, int, 'Plane']) -> Tuple[int, ...]:
        """
        Returns the remaining indices (0, 1, 2 excluding the value corresponding to the given plane).
        """
        plane_enum = cls._parse(plane)
        return tuple(i for i in range(3) if i != plane_enum.value)
