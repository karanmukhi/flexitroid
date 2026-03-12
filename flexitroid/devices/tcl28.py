from .general_der import GeneralDER, DERParameters
import numpy as np
import flexitroid.utils.device_sampling as sample
import cvxpy as cp



class TCLapprox(GeneralDER):
    def __init__(
        self,
        T: int,
        lmda: float,
        u_min: float,
        u_max: float,
        theta_min: float,
        theta_max: float,
        theta_init: float,
    ):
        assert u_min <= u_max, "Invalid power bounds"
        assert theta_min <= theta_max, "Invalid pre-departure SoC bounds"

        latent_temp = (
            np.ones(T) * theta_init * np.power(np.ones(T) * lmda, np.arange(1, T + 1))
        )
        u_min_arr = np.ones(T) * u_min
        u_max_arr = np.ones(T) * u_max
        # Set SoC bounds before departure
        theta_min_arr = theta_min - latent_temp
        theta_max_arr = theta_max - latent_temp

        self.u_min_arr = u_min_arr
        self.u_max_arr = u_max_arr
        self.theta_min_arr = theta_min_arr
        self.theta_max_arr = theta_max_arr
        self.lmda = lmda
        self.u_max = u_max
        self.u_min = u_min
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.theta_init = theta_init

        # Create power bound arrays with zeros outside charging window

        x_min_arr = np.array([self.p_interval(self, theta_init, t) for t in range(1,T+1)])
        x_max_arr = np.array([self.b_interval(self, theta_init, t) for t in range(1,T+1)])
        params = DERParameters(
            u_min=u_min_arr, u_max=u_max_arr, x_min=x_min_arr, x_max=x_max_arr
        )

        super().__init__(params)

    def A_b(self) -> np.ndarray:
        I, J = np.indices((self.T, self.T))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(self.T), -np.eye(self.T), x_A, -x_A])
        b = np.concatenate(
            [
                self.u_max_arr,
                -self.u_min_arr,
                self.theta_max_arr,
                -self.theta_min_arr,
            ]
        )
        return A, b

    def b_interval(self, x_0, t):
        raise NotImplementedError
    
    def p_interval(self, x_0, t):
        raise NotImplementedError
    
    def x_b(self, x_0, t):
        return min(
            self.theta_max,
            x_0 * np.power(self.lmda, t)
            + np.sum(np.power(self.lmda, np.arange(t))) * self.u_max,
        )

    def x_p(self, x_0, t):
        return max(self.theta_min, x_0 * np.power(self.lmda, t))
    
    def solve_exact_lp(self, c):
        A, b = self.A_b()
        c = c
        x = cp.Variable(self.T)
        constraints = [A @ x <= b]
        obj = cp.Minimize(c @ x)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return x.value, prob.value
    

class TCLinner(TCLapprox):
    def __init__(
        self,
        T: int,
        lmda: float,
        u_min: float,
        u_max: float,
        theta_min: float,
        theta_max: float,
        theta_init: float,
    ):
        self.b_interval = b_inner
        self.p_interval = p_inner
        super().__init__(
            T=T,
            lmda=lmda,
            u_min=u_min,
            u_max=u_max,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_init=theta_init,
        )

class TCLouter(TCLapprox):
    def __init__(
        self,
        T: int,
        lmda: float,
        u_min: float,
        u_max: float,
        theta_min: float,
        theta_max: float,
        theta_init: float,
    ):
        self.b_interval = b_outer
        self.p_interval = p_outer
        super().__init__(
            T=T,
            lmda=lmda,
            u_min=u_min,
            u_max=u_max,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_init=theta_init,
        )







def b_inner(self, x_0, t):
    assert x_0 <= self.theta_max, "Initial x_0 too high"
    if self.x_b(x_0, t) < self.theta_max:
        return self.u_max * t
    I, J = np.indices((t, t))
    x_A = np.tril(np.power(self.lmda, I - J))
    A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
    x_max = self.theta_max * np.ones(t - 1) - x_0 * np.power(self.lmda, np.arange(1, t))
    x_min = self.theta_min - x_0 * np.power(self.lmda, np.arange(1, t))

    b = np.concatenate(
        [
            self.u_max * np.ones(t),
            -self.u_min * np.ones(t),
            x_max,
            -x_min,
        ]
    )
    a = np.power(self.lmda, np.arange(t)[::-1])
    u = cp.Variable(t)
    constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_max]
    obj = cp.Minimize(cp.sum(u))
    prob = cp.Problem(obj, constraints)
    return prob.solve()

def p_inner(self, x_0, t):
    assert x_0 >= self.theta_min, "Initial x_0 too low"
    if self.x_p(x_0, t) > self.theta_min:
        return self.u_min * t
    I, J = np.indices((t, t))
    x_A = np.tril(np.power(self.lmda, I - J))
    A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
    x_max = self.theta_max * np.ones(t - 1) - x_0 * np.power(self.lmda, np.arange(1, t))
    x_min = self.theta_min - x_0 * np.power(self.lmda, np.arange(1, t))

    b = np.concatenate(
        [
            self.u_max * np.ones(t),
            -self.u_min * np.ones(t),
            x_max,
            -x_min,
        ]
    )
    a = np.power(self.lmda, np.arange(t)[::-1])
    u = cp.Variable(t)
    constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_min]
    obj = cp.Maximize(cp.sum(u))
    prob = cp.Problem(obj, constraints)
    return prob.solve()

def b_outer(self, x_0, t):
    assert x_0 <= self.theta_max, "Initial x_0 too high"

    if self.x_b(x_0, t) < self.theta_max:
        return self.u_max * t
    I, J = np.indices((t, t))
    x_A = np.tril(np.power(self.lmda, I - J))
    A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
    x_max = self.theta_max * np.ones(t - 1) - x_0 * np.power(self.lmda, np.arange(1, t))
    x_min = self.theta_min - x_0 * np.power(self.lmda, np.arange(1, t))

    b = np.concatenate(
        [
            self.u_max * np.ones(t),
            -self.u_min * np.ones(t),
            x_max,
            -x_min,
        ]
    )
    a = np.power(self.lmda, np.arange(t)[::-1])
    u = cp.Variable(t)
    constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_max]
    obj = cp.Maximize(cp.sum(u))
    prob = cp.Problem(obj, constraints)
    return prob.solve()

def p_outer(self, x_0, t):
    assert x_0 >= self.theta_min, "Initial x_0 too low"
    if self.x_p(x_0, t) > self.theta_min:
        return self.u_min * t
    I, J = np.indices((t, t))
    x_A = np.tril(np.power(self.lmda, I - J))
    A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
    x_max = self.theta_max * np.ones(t - 1) - x_0 * np.power(self.lmda, np.arange(1, t))
    x_min = self.theta_min - x_0 * np.power(self.lmda, np.arange(1, t))

    b = np.concatenate(
        [
            self.u_max * np.ones(t),
            -self.u_min * np.ones(t),
            x_max,
            -x_min,
        ]
    )
    a = np.power(self.lmda, np.arange(t)[::-1])
    u = cp.Variable(t)
    constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_min]
    obj = cp.Minimize(cp.sum(u))
    prob = cp.Problem(obj, constraints)
    return prob.solve()


def b_interval(self):
    t = 5
    x_0 = self.theta_min

    if self.x_b(x_0, t) < self.theta_max:
        return self.u_max * t
    I, J = np.indices((t, t))
    x_A = np.tril(np.power(self.lmda, I - J))
    A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
    x_max = self.theta_max * np.ones(t - 1) - x_0 * np.power(self.lmda, np.arange(1, t))
    x_min = self.theta_min - x_0 * np.power(self.lmda, np.arange(1, t))
    b = np.concatenate(
        [
            self.u_max * np.ones(t),
            -self.u_min * np.ones(t),
            x_max,
            -x_min,
        ]
    )
    a = np.power(self.lmda, np.arange(t)[::-1])
    u = cp.Variable(t)
    constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_max]
    obj = cp.Maximize(cp.sum(u))
    prob = cp.Problem(obj, constraints)
    return prob.solve()