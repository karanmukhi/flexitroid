"""Core aggregation module for DER flexibility sets.

This module implements the DER flexibility model and aggregation framework,
including individual flexibility sets and their Minkowski sums.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Optional, TypeVar, Generic
import numpy as np


@dataclass
class DERParameters:
    """Parameters defining a DER's flexibility.
    
    Args:
        u_min: Lower bound on power consumption for each timestep.
        u_max: Upper bound on power consumption for each timestep.
        x_min: Lower bound on state of charge for each timestep.
        x_max: Upper bound on state of charge for each timestep.
    """
    u_min: np.ndarray
    u_max: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    
    def __post_init__(self):
        """Validate parameter dimensions and constraints."""
        T = len(self.u_min)
        assert len(self.u_max) == T, "Power bounds must have same length"
        assert len(self.x_min) == T, "SoC bounds must have same length"
        assert len(self.x_max) == T, "SoC bounds must have same length"
        assert np.all(self.u_min <= self.u_max), "Invalid power bounds"
        assert np.all(self.x_min <= self.x_max), "Invalid SoC bounds"


class Device(ABC):
    """Abstract base class for all DER devices.
    
    This class defines the common interface that all DER devices must implement
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


class GeneralDER(Device):
    """General DER flexibility set representation.
    
    This class implements the individual flexibility set F(ξᵢ) for a single DER,
    defined by power and energy constraints.
    """
    
    def __init__(self, params: DERParameters):
        """Initialize the flexibility set.
        
        Args:
            params: DER parameters defining power and energy constraints.
        """
        self.params = params
        self._T = len(params.u_min)
        
    @property
    def T(self) -> int:
        return self._T
        
    def _sum_over_subset(self, A: Set[int], values: np.ndarray) -> float:
        """Helper function to sum values over a subset.
        
        Args:
            A: Subset of the ground set T.
            values: Array of values to sum over.
            
        Returns:
            Sum of values over the subset A.
        """
        return sum(values[i] for i in A)
    
    def _get_time_interval(self, t: int) -> Set[int]:
        """Get the time interval [t] = {0, 1, ..., t}.
        
        Args:
            t: End time of interval.
            
        Returns:
            Set of integers from 0 to t inclusive.
        """
        return set(range(t + 1))
    
    def b(self, A: Set[int]) -> float:
        """Compute submodular function b for the g-polymatroid representation.
        
        Args:
            A: Subset of the ground set T.
            
        Returns:
            Value of b(A) as defined by the recursive formula.
        """
        if not A:
            return 0.0
            
        t_max = max(A)
        T_t = self._get_time_interval(t_max)
        A_complement = T_t - A
        b = np.sum(self.u_u[list(A)])
        p_c = np.sum(self.u_l[list(A_c)])
        t_set = set()
        for t in range(t_max):
            t_set.add(t)
            b = np.min(
                [
                    b,
                    self.x_u[t]
                    - p_c
                    + np.sum(self.u_l[list(A_c - t_set)])
                    + np.sum(self.u_u[list(A - t_set)]),
                ]
            )
            p_c = np.max(
                [
                    p_c,
                    self.x_l[t]
                    - b
                    + np.sum(self.u_u[list(A - t_set)])
                    + np.sum(self.u_l[list(A_c - t_set)]),
                ]
            )
        return b
    
    
    def p(self, A: Set[int]) -> float:
        """Compute supermodular function p for the g-polymatroid representation.
        
        Args:
            A: Subset of the ground set T.
            
        Returns:
            Value of p(A) as defined by the recursive formula.
        """
        if not A:
            return 0.0
            
        t_max = max(A)
        T_t = self._get_time_interval(t_max)
        A_complement = T_t - A

        p = np.sum(self.u_l[list(A)])
        b_c = np.sum(self.u_u[list(A_c)])
        t_set = set()
        
        for t in range(t_max):
            t_set.add(t)
            p = np.max(
                [
                    p,
                    self.x_l[t]
                    - b_c
                    + np.sum(self.u_u[list(A_c - t_set)])
                    + np.sum(self.u_l[list(A - t_set)]),
                ]
            )
            b_c = np.min(
                [
                    b_c,
                    self.x_u[t]
                    - p
                    + np.sum(self.u_l[list(A - t_set)])
                    + np.sum(self.u_u[list(A_c - t_set)]),
                ]
            )
        return p


class PV(GeneralDER):
    """Photovoltaic system flexibility set representation.
    
    This class implements the flexibility set for a PV system with
    curtailment capabilities but no energy storage.
    """
    pass


class EV(GeneralDER):
    """Electric vehicle flexibility set representation.
    
    This class implements the flexibility set for an EV with
    bidirectional charging capabilities and battery storage.
    """
    pass


class ESS(GeneralDER):
    """Energy storage system flexibility set representation.
    
    This class implements the flexibility set for a stationary
    energy storage system with bidirectional power flow.
    """
    pass


D = TypeVar('D', bound=Device)

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
