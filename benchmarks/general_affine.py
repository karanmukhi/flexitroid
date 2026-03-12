import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
from benchmarks.benchmark import InnerApproximation
from flexitroid.utils.population_generator import PopulationGenerator


class GeneralAffine(InnerApproximation):
    def __init__(self, population: PopulationGenerator):
        name = "general_affine"
        super().__init__(name, population)

    def compute_A_b(self):
        N = self.N
        T = self.T

        bi = self.population.calculate_indiv_bs()
        hx = np.sum(bi, axis=1) / N  # shape: (4*T,)

        L = np.tril(np.ones((T, T)))
        Linv = np.linalg.inv(L)
        H = np.vstack([Linv, -Linv, np.eye(T), -np.eye(T)])

        Hi_list = [H for _ in range(N)]
        Hy = block_diag(*Hi_list)  # shape: (4*T*N, T*N)
        check_feasible(Hy, bi, T, N)

        P = np.zeros((T, T))
        pbar_ga = np.zeros(T)
        for h_i in bi.T:
            P_i, pbar_ga_i = general_affine_inner_approx(H, hx, h_i)
            P += P_i
            pbar_ga += pbar_ga_i

        A = H @ np.linalg.inv(P)
        b = hx + A @ pbar_ga
        A = A @ L
        return A, b


def check_feasible(Hy, hIs, T, N):
    Y = np.hstack([np.eye(T) for _ in range(N)])  # shape: (T, T*N)
    hy = hIs.flatten(order="F")  # shape: (4*T*N,)

    u = cp.Variable(T)
    ui = cp.Variable(T * N)

    objective = cp.Maximize(cp.sum(u))

    constraints = [Y @ ui == u, Hy @ ui <= hy]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)
    assert prob.status in [
        "optimal",
        "optimal_inaccurate",
    ], f"Infeasible problem: {prob.status}"


def calculate_indiv_sets(a, d, x_init, x_fin, u_max, u_min, x_max, N, T):
    """
    Generate hi vectors for flexibility sets {H*u <= hi}
    Assumes u = 0 for t < a and t > d.
    Inputs:
      a, d, x_init, x_fin, u_max, u_min, x_max: 1D numpy arrays of length N.
      N: number of EVs.
      T: number of time steps.
    Returns:
      hi: a (4*T x N) numpy array.
    """
    hi = np.zeros((4 * T, N))
    for i in range(N):
        # Create zero vectors of length T
        C_max = np.zeros(T)
        C_min = np.zeros(T)
        R_max = np.zeros(T)
        R_min = np.zeros(T)

        # Floor arrival and ceil departure (MATLAB uses 1-indexing; here we adjust accordingly)
        ai = int(np.floor(a[i]))
        di = int(np.ceil(d[i]))
        # In MATLAB, a(i):d(i) means indices from a[i] to d[i] (inclusive).
        # Here, since Python indexing starts at 0, we use indices ai-1 to di-1.
        R_max[ai:di] = u_max[i]
        R_min[ai:di] = u_min[i]
        C_max[ai:] = x_max[i] - x_init[i]
        # For C_min, first set indices corresponding to t from a to d-1
        if di - 1 > ai:
            C_min[ai : di - 1] = -x_init[i]
        C_min[di - 1 :] = x_fin[i] - x_init[i]

        hi[:, i] = np.concatenate((R_max, -R_min, C_max, -C_min))
    return hi


def general_affine_inner_approx(H, hx, hi):
    """
    Compute the maximum-volume general affine inner approximation given by
        p_bar + P*(base set)
    Inputs:
    Returns:
      (P, pbar) where P is a (n_x x n_x) matrix and pbar is a vector of length n_x.
    """
    T = H.shape[1]
    Y = np.eye(T)

    P_var = cp.Variable((T, T))
    pbar = cp.Variable(T)
    Gamma_i = cp.Variable((T, T))
    gamma_i = cp.Variable(T)
    Lambda_i = cp.Variable((4 * T, 4 * T), nonneg=True)

    constraints = [
        Y @ Gamma_i == P_var,
        Y @ gamma_i == pbar,
        Lambda_i @ H == H @ Gamma_i,
        Lambda_i @ hx <= hi - H @ gamma_i,
    ]

    objective = cp.Maximize(cp.trace(P_var))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)

    return P_var.value, pbar.value
