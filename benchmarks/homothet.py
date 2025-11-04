import cvxpy as cp
import numpy as np
from flexitroid.utils.population_generator import PopulationGenerator
from benchmarks.benchmark import InnerApproximation


class HomothetProjection(InnerApproximation):
    def __init__(self, population: PopulationGenerator):
        name = "homothet_projection"
        super().__init__(name, population)
        self.A_exact = self.get_A_exact()
        self.b_exact = self.get_b_exact()

    def get_A_exact(self):
        T = self.T
        return np.vstack(
            [-np.eye(T), np.eye(T), np.tril(np.ones((T, T))), -np.tril(np.ones((T, T)))]
        )

    def get_b_exact(self):
        T = self.T
        b_raw = self.population.calculate_indiv_bs().T
        u_max = b_raw[:, :T]
        minus_u_min = b_raw[:, T : 2 * T]
        x_max = b_raw[:, 2 * T : 3 * T]
        minus_x_min = b_raw[:, 3 * T : 4 * T]
        b = np.hstack([minus_u_min, u_max, x_max, minus_x_min]).T
        return b

    def compute_A_b(self):
        A = self.A_exact
        b = self.b_exact
        b_list = list(b.T)

        B, b_p = getAbProjection(A, b_list)
        """
        b_1 = np.ones(T)*(-np.mean(batts["x_min"]))
        b_2 = np.ones(T)*(np.mean(batts["x_max"]))
        b_3 = np.ones(T)*(np.mean(batts["S_max"])-np.mean(batts["S_0"]))/dt
        b_4 = np.ones(T)*(np.mean(batts["S_0"])/dt)
        H = np.concatenate((b_1,b_2,b_3,b_4))
        """
        H = np.mean(b_list, axis=0)

        beta, t = fitHomothetProjectionLinDescisionRule(A, H, B, b_p, self.T, self.N)
        b_approx = beta * H + A @ t
        A_approx = A
        return A_approx, b_approx


def getAbProjection(
    A, b_list
):  # gives the half-sapace representation of implicit M-sum
    A_barot_list = []
    A_barot = A
    for i in range(1, len(b_list)):
        A_barot = np.concatenate([A_barot, -A], axis=1)
    A_barot_list.append(A_barot)

    A_barot = np.concatenate([np.zeros([np.shape(A)[0], np.shape(A)[1]]), A], axis=1)
    A_barot = np.concatenate(
        [A_barot, np.zeros((np.shape(A)[0], (len(b_list) - 2) * np.shape(A)[1]))],
        axis=1,
    )
    A_barot_list.append(A_barot)
    for i in range(1, len(b_list) - 1):
        A_barot = np.zeros((np.shape(A)[0], (i + 1) * np.shape(A)[1]))
        A_barot = np.concatenate([A_barot, A], axis=1)
        A_barot = np.concatenate(
            [
                A_barot,
                np.zeros((np.shape(A)[0], (len(b_list) - i - 2) * np.shape(A)[1])),
            ],
            axis=1,
        )
        A_barot_list.append(A_barot)

    A_barot = A_barot_list[0]
    for i in range(1, len(A_barot_list)):
        A_barot = np.concatenate([A_barot, A_barot_list[i]], axis=0)

    b_barot = b_list[-1]
    for i in range(0, len(b_list) - 1):
        b_barot = np.concatenate([b_barot, b_list[i]])
    return A_barot, b_barot


def fitHomothetProjectionLinDescisionRule(F, H, B, c, T, N):
    # Setup dimensions
    I = np.eye(T)
    rows_B = 4 * T * N

    # Define variables
    s = cp.Variable(nonneg=True)  # scalar variable
    G = cp.Variable((rows_B, 4 * T), nonneg=True)
    r = cp.Variable(T)
    V = cp.Variable(T * N - T)
    aux = cp.Variable(rows_B)
    aux_IW = cp.Variable((T * N, T))
    aux_rV = cp.Variable(T * N)

    constraints = []

    # Constrain the top T rows of aux_IW to be the identity matrix
    for i in range(T):
        for j in range(T):
            constraints.append(aux_IW[i, j] == I[i, j])
    # The remaining rows of aux_IW are left unconstrained

    # Enforce that the first T entries of aux_rV equal r, and the rest equal -V
    constraints.append(aux_rV[:T] == r)
    constraints.append(aux_rV[T:] == -V)

    # For each i and j, enforce: G[i, :] @ F[:, j] == B[i, :] @ aux_IW[:, j]
    for i in range(rows_B):
        for j in range(T):
            constraints.append(G[i, :] @ F[:, j] == B[i, :] @ aux_IW[:, j])

    # Link aux with the weighted sum of H via G
    constraints.append(aux == G @ H)

    # Final constraint: aux <= s * c + B @ aux_rV
    constraints.append(aux <= s * c + B @ aux_rV)

    # Objective: minimize s
    objective = cp.Minimize(s)

    # Define and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(
        solver=cp.GUROBI
    )  # Optionally, pass a solver argument, e.g., solver=cp.GUROBI if available

    # Compute beta and offset vector t from the solution
    beta = 1 / s.value
    t_val = -r.value / s.value
    return beta, t_val



