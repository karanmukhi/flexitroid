"""Aggregator module for DER flexibility sets.

This module implements the aggregation framework for DER flexibility sets,
including the Minkowski sum of individual flexibility sets.
"""

from typing import List, Set, TypeVar, Generic
from flexitroid.flexitroid import Flexitroid
from flexitroid.devices.pv import PV
import numpy as np

D = TypeVar("D", bound=PV)


class PVAggregator(Flexitroid, Generic[D]):
    """Generic aggregator for device flexibility sets.

    This class implements the aggregate flexibility set F(Ξₙ) as the Minkowski
    sum of individual flexibility sets, represented as a g-polymatroid for a set of
    PV devices.
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

        u_min = np.zeros(self.T)
        u_max = np.zeros(self.T)
        for device in devices:
            u_min += device.params.u_min
            u_max += device.params.u_max
        self._u_min = u_min
        self._u_max = u_max

    @property
    def u_min(self) -> np.ndarray:
        return self._u_min

    @property
    def u_max(self) -> np.ndarray:
        return self._u_max

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
        return np.sum(self.u_max[list(A)])

    def p(self, A: Set[int]) -> float:
        """Compute supermodular function p for the g-polymatroid representation.

        Args:
            A: Subset of the ground set T.

        Returns:
            Value of p(A) as defined in Section II-D of the paper
        """
        return np.sum(self.u_min[list(A)])
