"""Aggregator module for DER flexibility sets.

This module implements the aggregation framework for DER flexibility sets,
including the Minkowski sum of individual flexibility sets.
"""

from typing import List, Set, TypeVar, Generic
from flexitroid.flexitroid import Flexitroid
from flexitroid.devices.level1 import V1G
import numpy as np

D = TypeVar("D", bound=V1G)


class V1GAggregator(Flexitroid, Generic[D]):
    """Generic aggregator for device flexibility sets.

    This class implements the aggregate flexibility set F(Ξₙ) as the Minkowski
    sum of individual flexibility sets, represented as a g-polymatroid for a set of
    V1G devices.
    """

    def __init__(self, devices: List[D]):
        """Initialize the aggregate flexibility set.

        Args:
            devices: List of devices to aggregate.
        """
        if not devices:
            raise ValueError("Must provide at least one device")

        self.devices = devices
        self._T = devices[0].T

        # Validate all devices have same time horizon
        for device in devices[1:]:
            if device.T != self.T:
                raise ValueError("All devices must have same time horizon")

        major = {}
        minor = {}
        for device in devices:
            a, d = device.a, device.d
            key = (a, d)
            major[key] = major.get(key, np.zeros(d - a)) + device.major
            minor[key] = minor.get(key, np.zeros(d - a)) + device.minor

        self._major = major
        self._minor = minor

    @property
    def major(self) -> np.ndarray:
        return self._major

    @property
    def minor(self) -> np.ndarray:
        return self._minor

    @property
    def T(self) -> int:
        return self._T

    def b(self, A: Set[int]) -> float:
        """Compute submodular function b for the g-polymatroid representation.

        Args:
            A: Subset of the ground set T.

        Returns:
            Value of b(A) as defined in Section II-D of the paper
        """
        return self.modular_extracted(self.major, A)

    def p(self, A: Set[int]) -> float:
        """Compute supermodular function p for the g-polymatroid representation.

        Args:
            A: Subset of the ground set T.

        Returns:
            Value of p(A) as defined in Section II-D of the paper
        """
        return self.modular_extracted(self.minor, A)

    def modular_extracted(self, agg_dict, A):
        result = 0
        for key in agg_dict.keys():
            active = set(range(key[0], key[1]))
            active_intersection = active.intersection(A)
            on_time = len(active_intersection)
            result += np.sum(agg_dict[key][:on_time])
        return result


class V1GConstrainted(V1GAggregator):
    def __init__(self, devices: List[D], a_l, a_u):
        self.a_l = a_l
        self.a_u = a_u
        super().__init__(devices)

    def p(self, A: Set[int]) -> float:
        res = super().p(A)
        return max(res, np.sum(self.a_l[list(A)]))
    
    def b(self, A: Set[int]) -> float:
        res = super().b(A)
        return max(res, np.sum(self.a_u[list(A)]))
