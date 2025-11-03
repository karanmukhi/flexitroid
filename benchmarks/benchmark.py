from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np
from flexitroid.utils.population_generator import PopulationGenerator
import time


class Benchmark(ABC):
    """Abstract base class for benchmarks.

    This class defines the common interface that all benchmark implementations
    must follow. It provides a standardized way to run benchmarks, collect metrics,
    and report results.
    """

    def __init__(self, name: str, population: PopulationGenerator):
        """Initialize the benchmark with a name."""
        self.name = name
        self.population = population
        self.T = population.T
        self.N = population.N

    @abstractmethod
    def solve_lp(self) -> None:
        pass

    @abstractmethod
    def solve_qp(self) -> None:
        pass

    @abstractmethod
    def solve_l_inf(self) -> None:
        pass


class InnerApproximation(Benchmark):
    """Abstract base class for benchmarks.

    This class defines the common interface that all benchmark implementations
    must follow. It provides a standardized way to run benchmarks, collect metrics,
    and report results.
    """

    def __init__(self, name: str, population: PopulationGenerator):
        """Initialize the benchmark with a name."""
        super().__init__(name, population)
        self.A = None
        self.b = None
        self.approximation_time = None

    @property
    def A_b(self):
        if self.A is None or self.b is None:
            start = time.perf_counter()
            self.A, self.b = self.compute_A_b()
            end = time.perf_counter()
            self.approximation_time = end - start
        return self.A, self.b

    @abstractmethod
    def compute_A_b(self):
        pass

    def reset(self):
        self.A = None
        self.b = None
        self.approximation_time = None

    def compute_unaggregatedA_b(self):
        ...

    def solve_lp(self, c, A=None, b=None) -> None:
        A_approx, b_approx = self.A_b
        if A is not None and b is not None:
            A = np.vstack([A, A_approx])
            b = np.hstack([b, b_approx])
        else:
            A = A_approx
            b = b_approx
        x = cp.Variable(self.T)
        constraints = [A @ x <= b]
        objective = cp.Minimize(c @ x)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI)
        self.lp = prob
        self.lp_x = x.value
        return prob

    def solve_qp(self, Q, c) -> None:
        A_approx, b_approx = self.A_b
        x = cp.Variable(self.T)
        constraints = [A_approx @ x <= b_approx]
        objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI)
        self.qp = prob
        self.qp_x = x.value
        return prob

    def solve_l_inf(self) -> None:
        A_approx, b_approx = self.A_b
        x = cp.Variable(self.T)
        t = cp.Variable(nonneg=True)
        constraints = [A_approx @ x <= b_approx, x <= t, -x <= t]
        objective = cp.Minimize(t)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI)
        self.l_inf = prob
        self.l_inf_x = x.value
        self.l_inf_t = t.value
        return prob
