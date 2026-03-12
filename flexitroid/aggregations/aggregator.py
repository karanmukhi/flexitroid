"""Aggregator module for DER flexibility sets.

This module implements the aggregation framework for DER flexibility sets,
including the Minkowski sum of individual flexibility sets.
"""

from typing import Set, TypeVar
from flexitroid.flexitroid import Flexitroid
from flexitroid.utils.population_generator import PopulationGenerator
from flexitroid.aggregations.pv_aggregator import PVAggregator
from flexitroid.aggregations.v1g_aggregator import V1GAggregator
import flexitroid.problems.signal_tracker as signal_tracker

import numpy as np


class Aggregator(Flexitroid):
    """Generic aggregator for device flexibility sets.

    This class implements the aggregate flexibility set F(Ξₙ) as the Minkowski
    sum of individual flexibility sets, represented as a g-polymatroid.
    """

    def __init__(self, population: PopulationGenerator):
        """Initialize the aggregate flexibility set.

        Args:
            fleet: List of fleet to aggregate.
        """
        if population.N == 0:
            raise ValueError("Must provide at least one device")

        self.population = population
        self._T = population.T
        self.fleet = self.group_devices()

    def group_devices(self):
        fleet = (
            self.population.device_groups["der"]
            + self.population.device_groups["e2s"]
            + self.population.device_groups["v2g"]
            + self.population.device_groups["tcl"]
        )
        pvs = self.population.device_groups["pv"]
        if len(pvs) > 0:
            pv_agg = PVAggregator(pvs)
            fleet.append(pv_agg)
        v1gs = (
            self.population.device_groups["v1g"] + self.population.device_groups["e1s"]
        )
        if len(v1gs) > 0:
            v1g_agg = V1GAggregator(v1gs)
            fleet.append(v1g_agg)

        return fleet
    
    def sample_constraints(self, tightness=1):
        prob = self.solve_l_inf()
        # dist = np.max(self.greedy(-np.arange(self._T))) - prob.value
        a_u = np.max(prob.solution) + np.ones(self._T) * tightness
        return a_u
    
    def solve_constrainted_lp(self, c, tightness=1):
        b = self.sample_constraints(tightness)
        A = np.eye(self._T)
        return self.solve_lp(c, A, b)
    
    def disaggregate(self, signal):
        problem = signal_tracker.SingalTracker(self, signal)
        problem.solve()
        pi = np.array(problem.PI)
        lmda = problem.lmda
        pi = pi[problem.lmda != 0]
        lmda = lmda[lmda != 0]

        disaggregation = []
        for device in self.population.device_list:
            u_i = np.zeros(self._T)
            for l, c in zip(lmda, pi):
                vertex = device.greedy(c)
                u_i += l*vertex
            disaggregation.append(u_i)
        return np.array(disaggregation)

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
