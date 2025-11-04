from .general_der import GeneralDER, DERParameters
import numpy as np
import flexitroid.utils.device_sampling as sample
import cvxpy as cp


class TCL(GeneralDER):
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

        # Create power bound arrays with zeros outside charging window
        u_min_arr = np.ones(T) * u_min
        u_max_arr = np.ones(T) * u_max

        x_min_arr = np.ones(T) * theta_min - theta_min
        x_max_arr = np.ones(T) * theta_max - theta_min

        params = DERParameters(
            u_min=u_min_arr, u_max=u_max_arr, x_min=x_min_arr, x_max=x_max_arr
        )

        latent_temp = (
            np.ones(T) * theta_init * np.power(np.ones(T) * lmda, np.arange(1, T + 1))
        )
        # Set SoC bounds before departure
        theta_min_arr = theta_min - latent_temp
        theta_max_arr = theta_max - latent_temp

        super().__init__(params)
        self.lmda = lmda
        self.u_max = u_max
        self.u_min = u_min
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.theta_init = theta_init
        self.u_min_arr = u_min_arr
        self.u_max_arr = u_max_arr
        self.theta_min_arr = theta_min_arr
        self.theta_max_arr = theta_max_arr

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

    def solve_exact_lp(self, c):
        A, b = self.A_b()
        x = cp.Variable(self.T)
        constraints = [A @ x <= b]
        obj = cp.Maximize(c @ x)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return x.value, prob.value

    def monte_carlo_sample(self, N):
        A, b = self.A_b()
        test = np.random.uniform(0, self.u_max, size=(N, self.T))
        test = test[np.all(A @ test.T - b[:, None] < 0, axis=0)]
        return test

    def get_mc_approx(self, N=1000):
        test = self.monte_carlo_sample(N)
        return sum([self.in_g_polymatroid_naive(v)[0] for v in test]) / len(test)

    def other_mc_approx(self, N=1000):
        A, b = self.A_b()
        test = np.random.uniform(0, self.u_max, size=(N, self.T))
        test = test[np.array([self.in_g_polymatroid_naive(v)[0] for v in test])]
        rest = test[np.all(A @ test.T - b[:, None] < 0, axis=0)]
        return len(rest) / len(test)

    @classmethod
    def example(cls, T: int = 24) -> "TCL":
        lmda, u_min, u_max, theta_min, theta_max, theta_init = sample_tcl(T)
        return cls(
            T=T,
            lmda=lmda,
            u_min=u_min,
            u_max=u_max,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_init=theta_init,
        )

    def split_into_consecutive_ranges(self, input_set):
        if len(input_set) == 0:
            return []
        sorted_list = sorted(input_set)
        result = []
        start = sorted_list[0]

        for i in range(1, len(sorted_list)):
            # Check if the current number is not consecutive to the previous
            if sorted_list[i] != sorted_list[i - 1] + 1:
                # Append a single number if it's a singleton, or a range otherwise
                if start == sorted_list[i - 1]:
                    result.append((start, start))
                else:
                    result.append((start, sorted_list[i - 1]))
                start = sorted_list[i]

        # Append the final range or singleton
        if start == sorted_list[-1]:
            result.append((start, start))
        else:
            result.append((start, sorted_list[-1]))

        return result

    def b(self, A: set) -> float:
        intervals = self.split_into_consecutive_ranges(A)

        b_tot = 0
        x = self.theta_init
        t_start = 0

        for I in intervals:
            t = I[0] - t_start
            x = self.x_p(x, t)
            t = I[1] - I[0] + 1
            b_tot += self.b_interval(x, t)
            x = self.x_b(x, t)
            t_start = I[1] + 1

        return b_tot

    def p(self, A: set) -> float:
        intervals = self.split_into_consecutive_ranges(A)

        p_tot = 0
        x = self.theta_init
        t_start = 0

        for I in intervals:
            t = I[0] - t_start
            x = self.x_b(x, t)
            t = I[1] - I[0] + 1
            p_tot += self.p_interval(x, t)
            x = self.x_p(x, t)
            t_start = I[1] + 1
        return p_tot

    def x_b(self, x_0, t):
        return min(
            self.theta_max,
            x_0 * np.power(self.lmda, t)
            + np.sum(np.power(self.lmda, np.arange(t))) * self.u_max,
        )

    def x_p(self, x_0, t):
        return max(self.theta_min, x_0 * np.power(self.lmda, t))

    def p_interval(self, x_0, t):
        raise NotImplementedError

    def b_interval(self, x_0, t):
        raise NotImplementedError


class TCLouter(TCL):
    def __init__(self, T, lmda, u_min, u_max, theta_min, theta_max, theta_init):
        super().__init__(
            T=T,
            lmda=lmda,
            u_min=u_min,
            u_max=u_max,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_init=theta_init,
        )

    def b_interval(self, x_0, t):
        if self.x_b(x_0, t) < self.theta_max:
            return self.u_max * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_max]
        obj = cp.Maximize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()

    def p_interval(self, x_0, t):
        assert x_0 >= self.theta_min, "Initial x_0 too low"
        if self.x_p(x_0, t) > self.theta_min:
            return self.u_min * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_min]
        obj = cp.Minimize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()


class TCLinner(TCL):
    def __init__(self, T, lmda, u_min, u_max, theta_min, theta_max, theta_init):
        super().__init__(
            T=T,
            lmda=lmda,
            u_min=u_min,
            u_max=u_max,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_init=theta_init,
        )

    def b(self, A: set) -> float:
        result = 0
        intervals = self.split_into_consecutive_ranges(A)

        x = self.theta_init
        init_p = 0

        x_prev = self.theta_init
        b_prev = 0
        init_b_prev = 0

        for I in intervals:

            init_b, fin_b = I

            fin_p = init_b - 1

            t_p = fin_p - init_p + 1
            x = self.x_p(x, t_p)
            # p = self.p_interval(x, t_p)
            p  = self.u_min * t_p
            b2 = self.b_interval(x_prev, fin_b - init_b_prev + 1) - p - b_prev

            t_b = fin_b - init_b + 1
            b1 = self.b_interval(x, t_b)

            x_prev = x
            b_prev = b1
            init_b_prev = init_b

            result += min(b1, b2)

            x = self.x_b(x, t_b)
            init_p = fin_b + 1
        return result
        # intervals = self.split_into_consecutive_ranges(A)

        # b_tot = 0
        # x = self.theta_init
        # t_start = 0

        # for I in intervals:
        #     t = I[0] - t_start
        #     x = self.x_p(x, t)
        #     t = I[1] - I[0] + 1
        #     b_tot += self.b_interval(x, t)
        #     x = self.x_b(x, t)
        #     t_start = I[1] + 1

        # return b_tot

    def p(self, A: set) -> float:
        result = 0
        intervals = self.split_into_consecutive_ranges(A)

        x = self.theta_init
        init_b = 0

        x_prev = self.theta_init
        p_prev = 0
        init_p_prev = 0

        for I in intervals:

            init_p, fin_p = I

            fin_b = init_p - 1

            t_b = fin_b - init_b + 1
            x = self.x_b(x, t_b)
            # p = self.b_interval(x, t_p)
            b = self.u_max * t_b
            p2 = self.p_interval(x_prev, fin_p - init_p_prev + 1) - b - p_prev

            t_p = fin_p - init_p + 1
            p1 = self.p_interval(x, t_p)

            x_prev = x
            p_prev = p1
            init_p_prev = init_p

            x = self.x_p(x, t_p)
            init_b = fin_p + 1
            result += max(p1, p2)
        return result

        # intervals = self.split_into_consecutive_ranges(A)

        # p_tot = 0
        # x = self.theta_init
        # t_start = 0

        # for I in intervals:
        #     t = I[0] - t_start
        #     x = self.x_b(x, t)
        #     t = I[1] - I[0] + 1
        #     p_tot += self.p_interval(x, t)
        #     x = self.x_p(x, t)
        #     t_start = I[1] + 1
        # return p_tot

    def b_interval(self, x_0, t):
        assert x_0 <= self.theta_max, "Initial x_0 too high"

        if self.x_b(x_0, t) < self.theta_max:
            return self.u_max * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_max]
        obj = cp.Minimize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()

    def p_interval(self, x_0, t):
        assert x_0 >= self.theta_min, "Initial x_0 too low"
        if self.x_p(x_0, t) > self.theta_min:
            return self.u_min * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_min]
        obj = cp.Maximize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()


def sample_tcl(T):
    lmda = np.random.uniform(0.95, 0.99)
    u_min = 0.
    u_max = np.random.uniform(4, 8)
    set_point = np.random.uniform(20, 25)
    deadband = np.random.uniform(1.5, 2.5)
    theta_max = set_point + deadband / 2
    theta_min = set_point - deadband / 2
    # theta_init = np.random.uniform(theta_min, theta_max)
    theta_init = theta_max
    return lmda, u_min, u_max, theta_min, theta_max, theta_init



class TCLplank(GeneralDER):
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


        x_min_arr = np.array([self.p_interval(theta_init, t) for t in range(1,T+1)])
        x_max_arr = np.array([self.b_interval(theta_init, t) for t in range(1,T+1)])
        # x_max_arr[-1] += 1e-2
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
        assert x_0 <= self.theta_max, "Initial x_0 too high"

        if self.x_b(x_0, t) < self.theta_max:
            return self.u_max * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_max]
        obj = cp.Minimize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()

    def p_interval(self, x_0, t):
        assert x_0 >= self.theta_min, "Initial x_0 too low"
        if self.x_p(x_0, t) > self.theta_min:
            return self.u_min * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_min]
        obj = cp.Maximize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()
    
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
        x = cp.Variable(self.T)
        constraints = [A @ x <= b]
        obj = cp.Maximize(c @ x)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return x.value, prob.value
    

from .general_der import GeneralDER, DERParameters
import numpy as np
import flexitroid.utils.device_sampling as sample
import cvxpy as cp


class TCL(GeneralDER):
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

        # Create power bound arrays with zeros outside charging window
        u_min_arr = np.ones(T) * u_min
        u_max_arr = np.ones(T) * u_max

        x_min_arr = np.ones(T) * theta_min - theta_min
        x_max_arr = np.ones(T) * theta_max - theta_min

        params = DERParameters(
            u_min=u_min_arr, u_max=u_max_arr, x_min=x_min_arr, x_max=x_max_arr
        )

        latent_temp = (
            np.ones(T) * theta_init * np.power(np.ones(T) * lmda, np.arange(1, T + 1))
        )
        # Set SoC bounds before departure
        theta_min_arr = theta_min - latent_temp
        theta_max_arr = theta_max - latent_temp

        super().__init__(params)
        self.lmda = lmda
        self.u_max = u_max
        self.u_min = u_min
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.theta_init = theta_init
        self.u_min_arr = u_min_arr
        self.u_max_arr = u_max_arr
        self.theta_min_arr = theta_min_arr
        self.theta_max_arr = theta_max_arr

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

    def solve_exact_lp(self, c):
        A, b = self.A_b()
        x = cp.Variable(self.T)
        constraints = [A @ x <= b]
        obj = cp.Maximize(c @ x)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return x.value, prob.value

    def monte_carlo_sample(self, N):
        A, b = self.A_b()
        test = np.random.uniform(0, self.u_max, size=(N, self.T))
        test = test[np.all(A @ test.T - b[:, None] < 0, axis=0)]
        return test

    def get_mc_approx(self, N=1000):
        test = self.monte_carlo_sample(N)
        return sum([self.in_g_polymatroid_naive(v)[0] for v in test]) / len(test)

    def other_mc_approx(self, N=1000):
        A, b = self.A_b()
        test = np.random.uniform(0, self.u_max, size=(N, self.T))
        test = test[np.array([self.in_g_polymatroid_naive(v)[0] for v in test])]
        rest = test[np.all(A @ test.T - b[:, None] < 0, axis=0)]
        return len(rest) / len(test)

    @classmethod
    def example(cls, T: int = 24) -> "TCL":
        lmda, u_min, u_max, theta_min, theta_max, theta_init = sample_tcl(T)
        return cls(
            T=T,
            lmda=lmda,
            u_min=u_min,
            u_max=u_max,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_init=theta_init,
        )

    def split_into_consecutive_ranges(self, input_set):
        if len(input_set) == 0:
            return []
        sorted_list = sorted(input_set)
        result = []
        start = sorted_list[0]

        for i in range(1, len(sorted_list)):
            # Check if the current number is not consecutive to the previous
            if sorted_list[i] != sorted_list[i - 1] + 1:
                # Append a single number if it's a singleton, or a range otherwise
                if start == sorted_list[i - 1]:
                    result.append((start, start))
                else:
                    result.append((start, sorted_list[i - 1]))
                start = sorted_list[i]

        # Append the final range or singleton
        if start == sorted_list[-1]:
            result.append((start, start))
        else:
            result.append((start, sorted_list[-1]))

        return result

    def b(self, A: set) -> float:
        intervals = self.split_into_consecutive_ranges(A)

        b_tot = 0
        x = self.theta_init
        t_start = 0

        for I in intervals:
            t = I[0] - t_start
            x = self.x_p(x, t)
            t = I[1] - I[0] + 1
            b_tot += self.b_interval(x, t)
            x = self.x_b(x, t)
            t_start = I[1] + 1

        return b_tot

    def p(self, A: set) -> float:
        intervals = self.split_into_consecutive_ranges(A)

        p_tot = 0
        x = self.theta_init
        t_start = 0

        for I in intervals:
            t = I[0] - t_start
            x = self.x_b(x, t)
            t = I[1] - I[0] + 1
            p_tot += self.p_interval(x, t)
            x = self.x_p(x, t)
            t_start = I[1] + 1
        return p_tot

    def x_b(self, x_0, t):
        return min(
            self.theta_max,
            x_0 * np.power(self.lmda, t)
            + np.sum(np.power(self.lmda, np.arange(t))) * self.u_max,
        )

    def x_p(self, x_0, t):
        return max(self.theta_min, x_0 * np.power(self.lmda, t))

    def p_interval(self, x_0, t):
        raise NotImplementedError

    def b_interval(self, x_0, t):
        raise NotImplementedError


class TCLouter(TCL):
    def __init__(self, T, lmda, u_min, u_max, theta_min, theta_max, theta_init):
        super().__init__(
            T=T,
            lmda=lmda,
            u_min=u_min,
            u_max=u_max,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_init=theta_init,
        )

    def b_interval(self, x_0, t):
        if self.x_b(x_0, t) < self.theta_max:
            return self.u_max * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_max]
        obj = cp.Maximize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()

    def p_interval(self, x_0, t):
        assert x_0 >= self.theta_min, "Initial x_0 too low"
        if self.x_p(x_0, t) > self.theta_min:
            return self.u_min * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_min]
        obj = cp.Minimize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()


class TCLinner(TCL):
    def __init__(self, T, lmda, u_min, u_max, theta_min, theta_max, theta_init):
        super().__init__(
            T=T,
            lmda=lmda,
            u_min=u_min,
            u_max=u_max,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_init=theta_init,
        )

    def b(self, A: set) -> float:
        result = 0
        intervals = self.split_into_consecutive_ranges(A)

        x = self.theta_init
        init_p = 0

        x_prev = self.theta_init
        b_prev = 0
        init_b_prev = 0

        for I in intervals:

            init_b, fin_b = I

            fin_p = init_b - 1

            t_p = fin_p - init_p + 1
            x = self.x_p(x, t_p)
            # p = self.p_interval(x, t_p)
            p  = self.u_min * t_p
            b2 = self.b_interval(x_prev, fin_b - init_b_prev + 1) - p - b_prev

            t_b = fin_b - init_b + 1
            b1 = self.b_interval(x, t_b)

            x_prev = x
            b_prev = b1
            init_b_prev = init_b

            result += min(b1, b2)

            x = self.x_b(x, t_b)
            init_p = fin_b + 1
        return result
        # intervals = self.split_into_consecutive_ranges(A)

        # b_tot = 0
        # x = self.theta_init
        # t_start = 0

        # for I in intervals:
        #     t = I[0] - t_start
        #     x = self.x_p(x, t)
        #     t = I[1] - I[0] + 1
        #     b_tot += self.b_interval(x, t)
        #     x = self.x_b(x, t)
        #     t_start = I[1] + 1

        # return b_tot

    def p(self, A: set) -> float:
        result = 0
        intervals = self.split_into_consecutive_ranges(A)

        x = self.theta_init
        init_b = 0

        x_prev = self.theta_init
        p_prev = 0
        init_p_prev = 0

        for I in intervals:

            init_p, fin_p = I

            fin_b = init_p - 1

            t_b = fin_b - init_b + 1
            x = self.x_b(x, t_b)
            # p = self.b_interval(x, t_p)
            b = self.u_max * t_b
            p2 = self.p_interval(x_prev, fin_p - init_p_prev + 1) - b - p_prev

            t_p = fin_p - init_p + 1
            p1 = self.p_interval(x, t_p)

            x_prev = x
            p_prev = p1
            init_p_prev = init_p

            x = self.x_p(x, t_p)
            init_b = fin_p + 1
            result += max(p1, p2)
        return result

        # intervals = self.split_into_consecutive_ranges(A)

        # p_tot = 0
        # x = self.theta_init
        # t_start = 0

        # for I in intervals:
        #     t = I[0] - t_start
        #     x = self.x_b(x, t)
        #     t = I[1] - I[0] + 1
        #     p_tot += self.p_interval(x, t)
        #     x = self.x_p(x, t)
        #     t_start = I[1] + 1
        # return p_tot

    def b_interval(self, x_0, t):
        assert x_0 <= self.theta_max, "Initial x_0 too high"

        if self.x_b(x_0, t) < self.theta_max:
            return self.u_max * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_max]
        obj = cp.Minimize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()

    def p_interval(self, x_0, t):
        assert x_0 >= self.theta_min, "Initial x_0 too low"
        if self.x_p(x_0, t) > self.theta_min:
            return self.u_min * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_min]
        obj = cp.Maximize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()


def sample_tcl(T):
    tau = 0.25
    lmda = np.exp(-4*tau / 25)
    u_min = 0.
    u_max = np.random.uniform(4, 8) * tau
    set_point = np.random.uniform(20, 25)
    deadband = np.random.uniform(1.5, 2.5)
    theta_max = set_point + deadband / 2
    theta_min = set_point - deadband / 2
    # theta_init = np.random.uniform(theta_min, theta_max)
    theta_init = theta_max
    return lmda, u_min, u_max, theta_min, theta_max, theta_init



class TCLplankDEV(GeneralDER):
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


        self.x_min_arr = np.array([self.p_interval(theta_init, t) for t in range(1,T+1)])
        self.x_max_arr = np.array([self.b_interval(theta_init, t) for t in range(1,T+1)])
        x_max_arr[-1] += 1e-2
        # params = DERParameters(
        #     u_min=u_min_arr, u_max=u_max_arr, x_min=x_min_arr, x_max=x_max_arr
        # )
        # super().__init__(params)

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
        assert x_0 <= self.theta_max, "Initial x_0 too high"

        if self.x_b(x_0, t) < self.theta_max:
            return self.u_max * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_max]
        obj = cp.Minimize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()

    def p_interval(self, x_0, t):
        assert x_0 >= self.theta_min, "Initial x_0 too low"
        if self.x_p(x_0, t) > self.theta_min:
            return self.u_min * t
        I, J = np.indices((t, t))
        x_A = np.tril(np.power(self.lmda, I - J))
        A = np.vstack([np.eye(t), -np.eye(t), x_A[:-1], -x_A[:-1]])
        b = np.concatenate(
            [
                self.u_max * np.ones(t),
                -self.u_min * np.ones(t),
                self.theta_max * np.ones(t - 1)
                - x_0 * np.power(self.lmda, np.arange(1, t)),
                x_0 * np.power(self.lmda, np.arange(1, t)) - self.theta_min,
            ]
        )
        a = np.power(self.lmda, np.arange(t)[::-1])
        u = cp.Variable(t)
        constraints = [A @ u <= b, a.T @ u + (self.lmda**t) * x_0 == self.theta_min]
        obj = cp.Maximize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()
    
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
        x = cp.Variable(self.T)
        constraints = [A @ x <= b]
        obj = cp.Maximize(c @ x)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return x.value, prob.value