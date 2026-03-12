import pytest
import numpy as np
import cvxpy as cp
from flexitroid.devices.level1 import V1G, E1S
from flexitroid.devices.level2 import V2G, E2S
from flexitroid.devices.pv import PV
from flexitroid.devices.general_der import GeneralDER


def test_pv_vertex():
    device = PV.example()
    device_vertex_tester(device)


def test_v1g_vertex():
    device = V1G.example()
    device_vertex_tester(device)


def test_v2g_vertex():
    device = V2G.example()
    device_vertex_tester(device)


def test_e1s_vertex():
    device = E1S.example()
    device_vertex_tester(device)


def test_e2s_vertex():
    device = E2S.example()
    device_vertex_tester(device)


def test_general_der_vertex():
    device = GeneralDER.example()
    device_vertex_tester(device)


def test_pv_top():
    device = PV.example()
    device_top_tester(device)


def test_v1g_top():
    device = V1G.example()
    device_top_tester(device)


def test_v2g_top():
    device = V2G.example()
    device_top_tester(device)


def test_e1s_top():
    device = E1S.example()
    device_top_tester(device)


def test_e2s_top():
    device = E2S.example()
    device_top_tester(device)


def test_general_der_top():
    device = GeneralDER.example()
    device_top_tester(device)


def test_pv_bottom():
    device = PV.example()
    device_bottom_tester(device)


def test_v1g_bottom():
    device = V1G.example()
    device_bottom_tester(device)


def test_v2g_bottom():
    device = V2G.example()
    device_bottom_tester(device)


def test_e1s_bottom():
    device = E1S.example()
    device_bottom_tester(device)


def test_e2s_bottom():
    device = E2S.example()
    device_bottom_tester(device)


def test_general_der_bottom():
    device = GeneralDER.example()
    device_bottom_tester(device)


def device_vertex_tester(device):
    T = device.T
    c = np.random.uniform(-1, 1, size=T)
    A, b = device.A_b()
    lp_sol = lp_solution(A, b, c)
    gp_sol = device.greedy(c)
    assert np.linalg.norm(gp_sol - lp_sol) < 1e-5


def device_top_tester(device):
    T = device.T
    c = np.random.uniform(0, 1, size=T)
    A, b = device.A_b()
    lp_sol = lp_solution(A, b, c)
    gp_sol = device.greedy(c)
    assert np.linalg.norm(gp_sol - lp_sol) < 1e-5


def device_bottom_tester(device):
    T = device.T
    c = np.random.uniform(-1, 0, size=T)
    A, b = device.A_b()
    lp_sol = lp_solution(A, b, c)
    gp_sol = device.greedy(c)
    assert np.linalg.norm(gp_sol - lp_sol) < 1e-5


def lp_solution(A, b, c):
    """Solve linear program to find optimal solution.
    Args:
        A: Matrix of constraints
        b: Vector of constraint bounds
        c: Vector of objective coefficients
    Returns:
        Optimal solution x
    """
    T = len(c)
    x = cp.Variable(T)
    constraints = [A @ x <= b]
    obj = cp.Maximize(-c.T @ x)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.GUROBI)
    return x.value
