import numpy as np
import cvxpy as cp
from flexitroid.devices.level1 import E1S
import flexitroid.problems.linear as lin
import flexitroid.problems.l_inf as linf
import flexitroid.problems.quadratic as qp


def test_LP(T=5):
    X = E1S.example(T)
    A = np.eye(T)
    b = np.random.uniform(size=T)
    c = -np.random.uniform(-1, 1, size=T)

    opt = solve_LP(X, A, b, c)
    prob = lin.LinearProgram(X, A, b, c)
    prob.solve()
    assert np.linalg.norm(prob.solution - opt) < 1e-5


def test_l_inf(T=5):
    X = E1S.example(T)

    opt = solve_l_inf(X)
    prob = linf.L_inf(X)
    prob.solve()
    assert np.linalg.norm(prob.solution - opt) < 1e-6


def test_QP(T=50):
    # TODO needs work on duality gap
    X = E1S.example(T)
    A, b = X.A_b()
    q = np.random.uniform(T)
    Q = q * np.eye(T)
    c = np.random.randn(T)
    x = solve_QP(Q, c, A, b)
    prob = qp.QuadraticProgram(X, Q, c, 1000)
    prob.solve()
    assert np.linalg.norm(prob.solution - x) < np.sum(x) / 100


def solve_LP(X, A, b, c):
    Ap, bp = X.A_b()
    Ap = np.vstack([Ap, A])
    bp = np.concatenate([bp, b])

    x = cp.Variable(X.T)
    prob = cp.Problem(cp.Minimize(c.T @ x), [Ap @ x <= bp])
    prob.solve(solver=cp.GUROBI)
    return x.value


def solve_QP(Q, c, A, b):
    T = c.shape[0]
    x = cp.Variable(T)
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)
    constraints = [A @ x <= b]
    problem = cp.Problem(objective, constraints)
    problem.solve()  # You can choose another solver if preferred.
    return x.value


def solve_l_inf(X):
    A, b = X.A_b()

    m, n = A.shape

    t = cp.Variable(nonneg=True)
    x = cp.Variable(n)
    objective = cp.Minimize(t)

    constraints = []
    constraints.append(A @ x <= b)
    constraints.append(x <= t)
    constraints.append(-x <= -t)
    primal = cp.Problem(objective, constraints)
    primal.solve()
    opt = x.value
    return opt
