from abc import ABC, abstractmethod
from typing import List, Set, Optional, TypeVar, Generic


class Flexitroid(ABC):
    """Abstract base class for flexiblity of DERs and aggregations of DERS.

    This class defines the common interface that flexibile entities must implement
    for flexibility set representation and computation.
    """

    @abstractmethod
    def b(self, A: Set[int]) -> float:
        """Compute submodular function b for the g-polymatroid representation."""
        pass

    @abstractmethod
    def p(self, A: Set[int]) -> float:
        """Compute supermodular function p for the g-polymatroid representation."""
        pass

    @property
    @abstractmethod
    def T(self) -> int:
        """Get the time horizon."""
        pass
