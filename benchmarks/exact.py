import numpy as np
import cvxpy as cp
from numerical_results.benchmarks.benchmark import Benchmark
from flexitroid.utils.population_generator import PopulationGenerator


class Exact(Benchmark):
    def __init__(self, population: PopulationGenerator):
        name = "exact"
        super().__init__(name, population)

    def solve_lp(self, c):
        T = self.T
        N = self.N

        Y = np.ones((T, N))

        As = self.population.calculate_indiv_As().T
        bs = self.population.calculate_indiv_bs().T

        ui = cp.Variable((N, T))

        constraints = [As[i] @ ui[i] <= bs[i] for i in range(N)]

        prob = cp.Problem(cp.Minimize(c @ Y @ ui), constraints)
        prob.solve(solver=cp.GUROBI)

        return prob

    def solve_qp(self, c, Q):
        T = self.T
        N = self.N
        Y = np.ones((T, N))

        As = self.population.calculate_indiv_As().T
        bs = self.population.calculate_indiv_bs().T

        ui = cp.Variable((N, T))
        u = cp.Variable(T)
        constratints = [As[i] @ ui[i] <= bs[i] for i in range(N)]

        objective = cp.Minimize(0.5 * cp.quad_form(Y @ ui, Q) + c @ Y @ ui)
        prob = cp.Problem(objective, constratints)
        prob.solve(solver=cp.GUROBI)
        return prob

    def solve_l_inf(self):
        T = self.T
        N = self.N

        Y = np.ones((T, N))

        As = self.population.calculate_indiv_As().T
        bs = self.population.calculate_indiv_bs().T

        ui = cp.Variable((N, T))
        t = cp.Variable(nonneg=True)
        constraints = []
        for i in range(N):
            constraints += [As[i] @ ui[i] <= bs[i]]
        constraints += [Y @ ui <= t, -Y @ ui <= t]
        objective = cp.Minimize(t)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI)
        return prob
