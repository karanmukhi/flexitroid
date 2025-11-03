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
    params = DERParameters(u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)
    return params


def pv(T):
    rated_power = U_MAX_BOUND * np.random.uniform()  # 5kW rated power
    # Create sinusoidal generation profile peaking at midday
    t = np.linspace(0, 2 * np.pi, T)
    base_profile = -np.maximum(
        0, np.random.uniform(0.1) + np.sin(t - np.pi / 2)
    )  # Negative = generation

    # Scale to realistic power bounds (kW)
    load = np.random.uniform(0.5, 1.5, size=T)
    u_min = rated_power * base_profile + load
    u_max = np.zeros_like(u_min) + load  # Can curtail to zero but not consume
    return u_min, u_max


def e1s(T):
    u_max = U_MAX_BOUND * np.random.uniform()
    x_max = X_MAX_BOUND * np.random.uniform()
    x_min = np.random.uniform(0, np.minimum(x_max, T * u_max))
    return u_max, x_min, x_max


def v1g(T):
    u_max = U_MAX_BOUND * np.random.uniform()

    a = np.random.randint(T - 1)
    d = np.random.randint(a + 2, T + 1)

    d = np.random.randint(a+2, T + 1)
    a = 0
    d = T
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

    a = 0
    d = T
    connected_time = d - a

    e_max = x_max
    e_min = np.random.uniform(0, np.minimum(e_max, connected_time * u_max))

    return a, d, u_min, u_max, x_min, x_max, e_min, e_max

def tcl(T):
    tau = 0.25
    R = np.random.uniform(2.2, 2.6)
    C = np.random.uniform(2.2, 2.6)
    lmda = np.exp(-tau / (R*C))
    u_min = 0.
    u_max = np.random.uniform(4, 8) * tau
    set_point = np.random.uniform(20, 25)
    deadband = np.random.uniform(1.5, 2.5)
    theta_max = set_point + deadband / 2
    theta_min = set_point - deadband / 2
    theta_init = np.random.uniform(theta_min, theta_max)
    return lmda, u_min, u_max, theta_min, theta_max, theta_init

# def tcl(T):
#     lmda = np.random.uniform(0.95,0.97)
#     lmda = 0.85
#     u_min = 0
#     u_max = np.random.uniform(4, 8)
#     set_point = np.random.uniform(20, 25)
#     deadband = np.random.uniform(1.5, 2.5)*5
#     theta_max = set_point + deadband / 2
#     theta_min = set_point - deadband / 2
#     theta_init = np.random.uniform(theta_min, theta_max)
#     return lmda, u_min, u_max, theta_min, theta_max, theta_init


def e2s(T):
    # Power parameters (kW)
    u_min = U_MIN_BOUND * np.random.uniform() * 1
    u_max = U_MAX_BOUND * np.random.uniform() * 1  # Can charge up to 2kW
    x_max = X_MAX_BOUND * np.random.uniform() * 1
    x_min = X_MIN_BOUND * np.random.uniform() * 1
    return u_min, u_max, x_min, x_max


@dataclass
class DERParameters:
    """Parameters defining a DER's flexibility.

    Args:
        u_min: Lower bound on power consumption for each timestep.
        u_max: Upper bound on power consumption for each timestep.
        x_min: Lower bound on state of charge for each timestep.
        x_max: Upper bound on state of charge for each timestep.
    """

    u_min: np.ndarray
    u_max: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray

    def __str__(self):
        return "sss"

    def __post_init__(self):
        """Validate parameter dimensions and constraints."""
        T = len(self.u_min)
        assert len(self.u_max) == T, "Power bounds must have same length"
        assert len(self.x_min) == T, "SoC bounds must have same length"
        assert len(self.x_max) == T, "SoC bounds must have same length"
        assert np.all(self.u_min <= self.u_max), "Invalid power bounds"
        assert np.all(self.x_min <= self.x_max), "Invalid SoC bounds"
