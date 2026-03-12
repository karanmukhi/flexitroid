"""Core aggregation module for DER flexibility sets.

This module implements the DER flexibility model and aggregation framework,
including individual flexibility sets and their Minkowski sums.
"""

from dataclasses import dataclass
from typing import Set
import numpy as np
import flexitroid.utils.device_sampling as sample
from flexitroid.utils.device_sampling import DERParameters
from flexitroid.flexitroid import Flexitroid
from flexitroid.cython.p_fast import p_fast
from flexitroid.cython.b_fast import b_fast


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
        self.active = set(range(self.T))

    @property
    def T(self) -> int:
        return self._T

    def A_b(self, remove_redundant=False) -> np.ndarray:
        A = np.vstack(
            [np.eye(self.T), -np.eye(self.T), np.tri(self.T), -np.tri(self.T)]
        )
        b = np.concatenate(
            [
                self.params.u_max,
                -self.params.u_min,
                self.params.x_max,
                -self.params.x_min,
            ]
        )
        if remove_redundant:
            A = A[np.isfinite(b)]
            b = b[np.isfinite(b)]
        return A, b

    def b(self, A: Set[int]) -> float:
        return b_fast(
            A,
            self.T,
            self.active,
            self.params.u_min,
            self.params.u_max,
            self.params.x_min,
            self.params.x_max,
        )

    def p(self, A: Set[int]) -> float:
        return p_fast(
            A,
            self.T,
            self.active,
            self.params.u_min,
            self.params.u_max,
            self.params.x_min,
            self.params.x_max,
        )

    def b_slow(self, A: Set[int]) -> float:
        A_c = self.active - A
        b = np.sum(self.params.u_max[list(A)])
        p_c = np.sum(self.params.u_min[list(A_c)])
        t_set = set()
        for t in range(self.T):
            t_set.add(t)
            b = np.min(
                [
                    b,
                    self.params.x_max[t]
                    - p_c
                    + np.sum(self.params.u_min[list(A_c - t_set)])
                    + np.sum(self.params.u_max[list(A - t_set)]),
                ]
            )
            p_c = np.max(
                [
                    p_c,
                    self.params.x_min[t]
                    - b
                    + np.sum(self.params.u_max[list(A - t_set)])
                    + np.sum(self.params.u_min[list(A_c - t_set)]),
                ]
            )
        return b

    def p_slow(self, A: Set[int]) -> float:
        A_c = self.active - A
        p = np.sum(self.params.u_min[list(A)])
        b_c = np.sum(self.params.u_max[list(A_c)])
        t_set = set()
        for t in range(self.T):
            t_set.add(t)
            p = np.max(
                [
                    p,
                    self.params.x_min[t]
                    - b_c
                    + np.sum(self.params.u_max[list(A_c - t_set)])
                    + np.sum(self.params.u_min[list(A - t_set)]),
                ]
            )
            b_c = np.min(
                [
                    b_c,
                    self.params.x_max[t]
                    - p
                    + np.sum(self.params.u_min[list(A - t_set)])
                    + np.sum(self.params.u_max[list(A_c - t_set)]),
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
        params = sample.der(T)
        return cls(params)
