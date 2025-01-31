import pytest
import numpy as np
import cvxpy as cp
from flexitroid.devices.level1 import V1G, E1S
from flexitroid.devices.level2 import V2G, E2S
from flexitroid.devices.generation import PV
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


def device_vertex_tester(device):
    T = device.T
    c = np.random.uniform(size=T)
    A, b = device.A_b
    lp_sol = lp_solution(A, b, c)
    gp_sol = device.solve_linear_program(c)
    assert (
        np.linalg.norm(gp_sol - lp_sol) < 1e-3
    )  # Increased tolerance for numerical stability


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
    prob.solve()
    return x.value
