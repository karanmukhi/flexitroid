import numpy as np
import cvxpy as cp
from flexitroid.utils.population_generator import PopulationGenerator
from benchmarks.benchmark import InnerApproximation


class Zonotope(InnerApproximation):
    def __init__(self, population: PopulationGenerator):
        name = "zonotope"
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
        T = self.T
        Z, G = generateZonotope(T, [0] * T)
        C = getMatrixC(T)  # calculate matrix of half-space representation Cx<=d
        Zonotope_list = []
        for b in b_list:
            d_new = getHyperplaneOffset(
                A, C, b, T
            )  # calculate vector of half-space representation Cx<=d
            Z = optimalZonotopeMaxNorm(
                A, b, G, C, d_new
            )  # calulate optimal center and scaling limits
            Zonotope_list.append(Z)

        # Calculate M-sum of zonotopes
        Zonotope_minkowski_list = []
        for l in range(len(Zonotope_list[0])):
            s = np.array(Zonotope_list[0][l])
            for h in range(1, len(Zonotope_list)):
                s = s + np.array(Zonotope_list[h][l])
            Zonotope_minkowski_list.append(list(s))

        b_approx = getVectord(C, Zonotope_minkowski_list, T)
        A_approx = C
        return A_approx, b_approx


def generateZonotope(
    T, c
):  # generates Zonotope Z(c,g_i,...g_p) and matrix of generators G
    G1 = np.eye(T)
    G2_1 = -1 / np.sqrt(2) * np.eye(T, T - 1)
    G2_2 = 1 / np.sqrt(2) * np.eye(T, T - 1, -1)
    G2 = G2_1 + G2_2
    G = np.column_stack((G1, G2))

    Z = [c]
    for i in range(np.shape(G)[1]):
        Z.append(list(G[:, i]))
    return Z, G


def getMatrixC(T):  # calculates matrix of half-space representation
    N = np.eye(T)
    Bl = np.tril(np.ones([T, T]))

    for i in range(1, T):
        block = Bl[i, 0 : i + 1] / np.linalg.norm(Bl[i, 0 : i + 1])
        Nblock = np.zeros([T - i, T])

        for j in range(0, T - i):
            Nblock[j, j : j + i + 1] = block

        N = np.concatenate([N, Nblock], axis=0)
    return np.concatenate([N, -N], axis=0)


def getHyperplaneOffset(
    A, C, b, dimension
):  # calculates vector of half-space representation
    m = np.shape(C)[0]
    d_list = []
    for i in range(m):
        x = cp.Variable(dimension)
        objective = cp.Maximize(C[i, :] @ x)
        constraints = [A @ x <= b]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI)
        d_list.append(C[i, :] @ x.value)
    return d_list


def optimalZonotopeMaxNorm(
    A, b, G, F, bi_list
):  # calculates optimal optimal center and vector of scaling limits, returns Z(c,g_i,...,g_p)
    AG = np.abs(A @ G)
    W = np.abs(F[0 : int(np.shape(F)[0] / 2)] @ G)
    delta_p = np.array(bi_list)
    W_aux = np.row_stack((W, W))
    t = cp.Variable(nonneg=True)
    c = cp.Variable(np.shape(A)[1])
    beta_bar = cp.Variable(np.shape(G)[1], nonneg=True)
    objective = cp.Minimize(t)
    constraints = []
    for i in range(len(delta_p)):
        constraints.append(-t <= delta_p[i] - (F[i, :] @ c + W_aux[i, :] @ beta_bar))
        constraints.append(delta_p[i] - (F[i, :] @ c + W_aux[i, :] @ beta_bar) <= t)
    constraints.append(AG @ beta_bar + A @ c <= b)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)
    c = c.value
    beta_bar = beta_bar.value

    Z = [list(c)]
    for i in range(np.shape(G)[1]):
        Z.append(list(G[:, i] * beta_bar[i]))
    return Z


def getVectord(
    C, Z, T
):  # calculates half-space representation vector from Z(c_i,g_i,...,g_p) representation
    p = 2 * T - 1  # number of generating directions
    c = Z[0]
    G_list = Z[1:]
    C = C[0 : int(np.shape(C)[0] / 2), :]
    delta_d_list = []
    for j in range(np.shape(C)[0]):
        delta_d = 0
        for i in range(p):
            delta_d = delta_d + np.abs(C[j, :] @ np.array(G_list[i]))
        delta_d_list.append(delta_d)
    d_list = []
    for i in range(np.shape(C)[0]):
        d_list.append(C[i, :] @ np.array(c) + np.array(delta_d_list[i]))
    for i in range(np.shape(C)[0]):
        d_list.append(-C[i, :] @ np.array(c) + np.array(delta_d_list[i]))
    d = np.array(d_list)
    return d

def get_G0(T):
    I = np.eye(T)
    q = np.zeros(T)
    r = np.array([-1,1])
    r = r / np.linalg.norm(r)
    q[:2] = r
    J = np.vstack([np.roll(q, i) for i in range(0,T-1)]).T
    return np.hstack([I, J])


def compute_A_b(A_list, b_list):
    T = A_list[0].shape[1]
    Z, G = generateZonotope(T, [0] * T)
    C = getMatrixC(T)  # calculate matrix of half-space representation Cx<=d
    Zonotope_list = []
    for A, b in zip(A_list, b_list):
        d_new = getHyperplaneOffset(
            A, C, b, T
        )  # calculate vector of half-space representation Cx<=d
        Z = optimalZonotopeMaxNorm(
            A, b, G, C, d_new
        )  # calulate optimal center and scaling limits
        Zonotope_list.append(Z)

    # Calculate M-sum of zonotopes
    Zonotope_minkowski_list = []
    for l in range(len(Zonotope_list[0])):
        s = np.array(Zonotope_list[0][l])
        for h in range(1, len(Zonotope_list)):
            s = s + np.array(Zonotope_list[h][l])
        Zonotope_minkowski_list.append(list(s))

    b_approx = getVectord(C, Zonotope_minkowski_list, T)
    A_approx = C
    return A_approx, b_approx