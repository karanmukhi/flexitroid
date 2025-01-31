"""Core aggregation module for DER flexibility sets.

This module implements the DER flexibility model and aggregation framework,
including individual flexibility sets and their Minkowski sums.
"""

from dataclasses import dataclass
from typing import Set
import numpy as np

from flextroid.flexitroid import Flexitroid
from . import modular_functions as mf


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
