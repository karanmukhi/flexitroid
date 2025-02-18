import numpy as np
from dataclasses import dataclass

U_MAX_BOUND = 1
U_MIN_BOUND = -1
X_MAX_BOUND = 10
X_MIN_BOUND = -10
assert U_MAX_BOUND > U_MIN_BOUND
assert X_MAX_BOUND > X_MIN_BOUND


def der(T):
    u_min = U_MIN_BOUND * np.random.uniform(size=T)
    u_max = U_MAX_BOUND * np.random.uniform(size=T)  # Can charge up to 2kW
    x_max = X_MAX_BOUND * np.random.uniform(size=T)
    x_min = X_MIN_BOUND * np.random.uniform(size=T)
    return u_min, u_max, x_min, x_max


def pv(T):
    rated_power = U_MAX_BOUND * np.random.uniform()  # 5kW rated power
    # Create sinusoidal generation profile peaking at midday
    t = np.linspace(0, 2 * np.pi, T)
    base_profile = -np.maximum(
        0, np.random.uniform(0.1) + np.sin(t - np.pi / 2)
    )  # Negative = generation

    # Scale to realistic power bounds (kW)
    u_min = rated_power * base_profile
    u_max = np.zeros_like(u_min)  # Can curtail to zero but not consume
    return u_min, u_max


def e1s(T):
    u_max = U_MAX_BOUND * np.random.uniform()
    x_max = X_MAX_BOUND * np.random.uniform()
    x_min = 0
    return u_max, x_min, x_max


def v1g(T):
    u_max = U_MAX_BOUND * np.random.uniform()
    a = np.random.randint(T - 1)
    d = np.random.randint(a + 1, T + 1)

    connected_time = d - a

    e_max = connected_time * u_max * np.random.uniform()
    e_min = e_max * np.random.uniform()

    return a, d, u_max, e_min, e_max


def v2g(T):
    u_min = U_MIN_BOUND * np.random.uniform()
    u_max = U_MAX_BOUND * np.random.uniform()
    x_max = X_MAX_BOUND * np.random.uniform()
    x_min = 0

    # Timing parameters
    a = np.random.randint(T - 1)
    d = np.random.randint(a + 1, T + 1)

    # a = 0
    # d = T
    connected_time = d - a

    e_max = x_max
    e_min = np.random.uniform(0, np.minimum(e_max, connected_time * u_max))

    return a, d, u_min, u_max, x_min, x_max, e_min, e_max


def e2s(T):
    # Power parameters (kW)
    u_min = U_MIN_BOUND * np.random.uniform()
    u_max = U_MAX_BOUND * np.random.uniform()  # Can charge up to 2kW
    x_max = X_MAX_BOUND * np.random.uniform()
    x_min = X_MIN_BOUND * np.random.uniform()
    return u_min, u_max, x_min, x_max
