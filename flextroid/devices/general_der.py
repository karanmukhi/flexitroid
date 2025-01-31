"""Core aggregation module for DER flexibility sets.

This module implements the DER flexibility model and aggregation framework,
including individual flexibility sets and their Minkowski sums.
"""

from dataclasses import dataclass
from typing import Set
import numpy as np

from flextroid.flexitroid import Flexitroid


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


class GeneralDER(Flexitroid):
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
        A_c = T_t - A
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
        A_c = T_t - A

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

    @classmethod
    def example(cls, T: int = 24) -> "GeneralDER":
        """Create an example DER with typical power and energy constraints.
        
        Creates a DER with:
        - Bidirectional power flow (-2kW to 2kW)
        - Energy storage capacity of 10kWh
        - Must maintain state of charge between 20% and 80%
        
        Args:
            T: Number of timesteps (default 24 for hourly resolution)
            
        Returns:
            GeneralDER instance with example parameters
        """
        # Power bounds (kW)
        u_min = np.full(T, -2.0)  # Can discharge up to 2kW
        u_max = np.full(T, 2.0)   # Can charge up to 2kW
        
        # Energy bounds (kWh)
        capacity = 10.0  # 10kWh battery
        soc_min = 0.2   # 20% minimum state of charge
        soc_max = 0.8   # 80% maximum state of charge
        
        x_min = np.full(T, capacity * soc_min)  # 2kWh minimum
        x_max = np.full(T, capacity * soc_max)  # 8kWh maximum
        
        params = DERParameters(
            u_min=u_min,
            u_max=u_max,
            x_min=x_min,
            x_max=x_max
        )
        return cls(params)
