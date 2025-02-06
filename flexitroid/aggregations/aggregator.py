"""Aggregator module for DER flexibility sets.

This module implements the aggregation framework for DER flexibility sets,
including the Minkowski sum of individual flexibility sets.
"""

from typing import List, Set, TypeVar, Generic
from flexitroid.flexitroid import Flexitroid

D = TypeVar("D", bound=Flexitroid)


class Aggregator(Flexitroid, Generic[D]):
    """Generic aggregator for device flexibility sets.

    This class implements the aggregate flexibility set F(Ξₙ) as the Minkowski
    sum of individual flexibility sets, represented as a g-polymatroid.
    """

    def __init__(self, fleet: List[D]):
        """Initialize the aggregate flexibility set.

        Args:
            fleet: List of fleet to aggregate.
        """
        if not fleet:
            raise ValueError("Must provide at least one device")

        self.fleet = fleet
        self._T = fleet[0].T

        # Validate all fleet have same time horizon
        for device in fleet[1:]:
            if device.T != self.T:
                raise ValueError("All fleet must have same time horizon")

    @property
    def T(self) -> int:
        return self._T

    def b(self, A: Set[int]) -> float:
        """Compute aggregate submodular function b.

        Args:
            A: Subset of the ground set T.

        Returns:
            Sum of individual b functions over all fleet.
        """
        return sum(device.b(A) for device in self.fleet)

    def p(self, A: Set[int]) -> float:
        """Compute aggregate supermodular function p.

        Args:
            A: Subset of the ground set T.

        Returns:
            Sum of individual p functions over all fleet.
        """
        return sum(device.p(A) for device in self.fleet)
