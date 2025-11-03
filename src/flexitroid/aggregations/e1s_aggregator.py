"""Aggregator module for DER flexibility sets.

This module implements the aggregation framework for DER flexibility sets,
including the Minkowski sum of individual flexibility sets.
"""

from typing import List, Set, TypeVar, Generic
from flexitroid.flexitroid import Flexitroid
from flexitroid.devices.level1 import E1S
import numpy as np

D = TypeVar("D", bound=E1S)


class E1SAggregator(Flexitroid, Generic[D]):
    """Generic aggregator for device flexibility sets.

    This class implements the aggregate flexibility set F(Ξₙ) as the Minkowski
    sum of individual flexibility sets, represented as a g-polymatroid for a set of
    E1S devices.
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

        major = np.zeros(self.T)
        minor = np.zeros(self.T)
        for device in devices:
            major += device.major
            minor += device.minor
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
        on_time = len(A)
        return np.sum(self.major[:on_time])

    def p(self, A: Set[int]) -> float:
        """Compute supermodular function p for the g-polymatroid representation.

        Args:
            A: Subset of the ground set T.

        Returns:
            Value of p(A) as defined in Section II-D of the paper
        """
        on_time = len(A)
        return np.sum(self.minor[:on_time])
