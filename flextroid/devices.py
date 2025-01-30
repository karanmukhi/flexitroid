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
        b = np.sum(self.u_max[list(A)])
        p_c = np.sum(self.u_min[list(A_c)])
        t_set = set()
        for t in range(t_max):
            t_set.add(t)
            b = np.min(
                [
                    b,
                    self.x_max[t]
                    - p_c
                    + np.sum(self.u_min[list(A_c - t_set)])
                    + np.sum(self.u_max[list(A - t_set)]),
                ]
            )
            p_c = np.max(
                [
                    p_c,
                    self.x_min[t]
                    - b
                    + np.sum(self.u_max[list(A - t_set)])
                    + np.sum(self.u_min[list(A_c - t_set)]),
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
    def __init__(self, u_min: np.ndarray, u_max: np.ndarray):
        """Initialize PV flexibility set with power bounds only.
        
        Args:
            u_min: Lower bound on power consumption for each timestep.
            u_max: Upper bound on power consumption for each timestep.
        """
        T = len(u_min)
        assert len(u_max) == T, "Power bounds must have same length"
        assert np.all(u_min <= u_max), "Invalid power bounds"
        
        # Create DER parameters with infinite energy bounds
        params = DERParameters(
            u_min=u_min,
            u_max=u_max,
            x_min=np.full(T, -np.inf),
            x_max=np.full(T, np.inf)
        )
        super().__init__(params)


class V2G(GeneralDER):
    """Vehicle-to-Grid Level 2 flexibility set representation.
    
    This class implements the flexibility set for an EV with
    bidirectional charging capabilities (V2G) and battery storage.
    
    The V2G has time-dependent constraints based on arrival and departure times:
    - Power bounds are 0 outside the charging window (before arrival, after departure)
    - State of charge has different bounds before and after departure, to reflect consumer requirements when the vehicle departs
    - Supports bidirectional power flow (both charging and discharging)
    """
    def __init__(self, T: int, a: int, d: int, 
                 u_min: float, u_max: float,
                 x_min: float, x_max: float,
                 e_min: float, e_max: float):
        """Initialize EV flexibility set with time-dependent constraints.
        
        Args:
            T: Time horizon length
            a: Arrival time
            d: Departure time
            u_min: Minimum power consumption during charging window
            u_max: Maximum power consumption during charging window
            x_min: Minimum state of charge before departure
            x_max: Maximum state of charge before departure
            e_min: Minimum state of charge after departure
            e_max: Maximum state of charge after departure
        """
        assert 0 <= a < d <= T, "Invalid arrival/departure times"
        assert u_min <= u_max, "Invalid power bounds"
        assert x_min <= x_max, "Invalid pre-departure SoC bounds"
        assert e_min <= e_max, "Invalid post-departure SoC bounds"
        
        # Create power bound arrays with zeros outside charging window
        u_min_arr = np.zeros(T)
        u_max_arr = np.zeros(T)
        u_min_arr[a:d] = u_min
        u_max_arr[a:d] = u_max
        
        # Create SoC bound arrays with different constraints before/after departure
        x_min_arr = np.full(T, -np.inf)  # Initialize with no constraints
        x_max_arr = np.full(T, np.inf)   # Initialize with no constraints
        
        # Set SoC bounds before departure
        x_min_arr[:d] = x_min
        x_max_arr[:d] = x_max
        
        # Set SoC bounds after departure
        x_min_arr[d:] = e_min
        x_max_arr[d:] = e_max
        
        # Initialize parent class with constructed parameter arrays
        params = DERParameters(
            u_min=u_min_arr,
            u_max=u_max_arr,
            x_min=x_min_arr,
            x_max=x_max_arr
        )
        super().__init__(params)


class V1G(GeneralDER):
    """Vehicle-to-Grid Level 1 flexibility set representation.
    
    This class implements the flexibility set for an EV with
    unidirectional charging capabilities (charging only, no discharging)
    and battery storage.
    
    The V1G has time-dependent constraints based on arrival and departure times:
    - Power bounds are 0 outside the charging window (before arrival, after departure)
    - State of charge has different bounds before and after departure
    - Only allows positive power flow (charging only)
    """
    def __init__(self, T: int, a: int, d: int, 
                 u_max: float,
                 e_min: float, e_max: float):
        """Initialize V1G flexibility set with time-dependent constraints.
        
        Args:
            T: Time horizon length
            a: Arrival time
            d: Departure time
            u_max: Maximum power consumption during charging window
            x_min: Minimum state of charge before departure
            x_max: Maximum state of charge before departure
            e_min: Minimum state of charge after departure
            e_max: Maximum state of charge after departure
        """
        assert 0 <= a < d <= T, "Invalid arrival/departure times"
        assert 0 <= u_max, "Invalid power bounds (must be non-negative)"
        assert e_min <= e_max, "Invalid post-departure SoC bounds"
        
        # Create power bound arrays with zeros outside charging window
        u_min_arr = np.zeros(T)
        u_max_arr = np.zeros(T)
        u_min_arr[a:d] = u_min
        u_max_arr[a:d] = u_max
        
        # Create SoC bound arrays with different constraints before/after departure
        x_min_arr = np.full(T, -np.inf)  # Initialize with no constraints
        x_max_arr = np.full(T, np.inf)   # Initialize with no constraints
        
        # Set SoC bounds after departure
        x_min_arr[d:] = e_min
        x_max_arr[d:] = e_max
        
        # Initialize parent class with constructed parameter arrays
        params = DERParameters(
            u_min=u_min_arr,
            u_max=u_max_arr,
            x_min=x_min_arr,
            x_max=x_max_arr
        )
        super().__init__(params)


class ESS(GeneralDER):
    """Energy storage system flexibility set representation.
    
    This class implements the flexibility set for a stationary
    energy storage system with bidirectional power flow.
    """
    def __init__(self, u_min: float, u_max: float, x_min: float, x_max: float, T: int):
        """Initialize ESS flexibility set with constant power and energy bounds.
        
        Args:
            u_min: Lower bound on power consumption (constant over time).
            u_max: Upper bound on power consumption (constant over time).
            x_min: Lower bound on state of charge (constant over time).
            x_max: Upper bound on state of charge (constant over time).
            T: Time horizon length.
        """
        assert u_min <= u_max, "Invalid power bounds"
        assert x_min <= x_max, "Invalid energy bounds"
        
        # Create DER parameters with constant bounds
        params = DERParameters(
            u_min=np.full(T, u_min),
            u_max=np.full(T, u_max),
            x_min=np.full(T, x_min),
            x_max=np.full(T, x_max)
        )
        super().__init__(params)
