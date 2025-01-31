"""Aggregator module for DER flexibility sets.

This module implements the aggregation framework for DER flexibility sets,
including the Minkowski sum of individual flexibility sets.
"""

from typing import List, Set, TypeVar, Generic
from .aggregation import Device

D = TypeVar("D", bound=Device)


class Aggregator(Generic[D]):
    """Generic aggregator for device flexibility sets.

    This class implements the aggregate flexibility set F(Ξₙ) as the Minkowski
    sum of individual flexibility sets, represented as a g-polymatroid.
    """

    def __init__(self, devices: List[D]):
        """Initialize the aggregate flexibility set.

        Args:
            devices: List of devices to aggregate.
        """
        if not devices:
            raise ValueError("Must provide at least one device")

        self.devices = devices
        self.T = devices[0].T

        # Validate all devices have same time horizon
        for device in devices[1:]:
            if device.T != self.T:
                raise ValueError("All devices must have same time horizon")

    def b(self, A: Set[int]) -> float:
        """Compute aggregate submodular function b.

        Args:
            A: Subset of the ground set T.

        Returns:
            Sum of individual b functions over all devices.
        """
        return sum(device.b(A) for device in self.devices)

    def p(self, A: Set[int]) -> float:
        """Compute aggregate supermodular function p.

        Args:
            A: Subset of the ground set T.

        Returns:
            Sum of individual p functions over all devices.
        """
        return sum(device.p(A) for device in self.devices)
