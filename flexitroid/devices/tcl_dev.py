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

        # x_max_arr = np.min([theta_max * np.ones(T), theta_max*np.power(lmda, np.arange(1,T+1)) + np.cumsum(np.pow(lmda, np.arange(T))) * u_max], axis=0)
        # x_min_arr = np.max([theta_min * np.ones(T), theta_min*np.power(lmda, np.arange(1,T+1)) + np.cumsum(np.pow(lmda, np.arange(T))) * u_min], axis=0)

        # x_min_arr = theta_min * (1-lmda) * np.arange(1,T+1) + theta_min - theta_init
        # x_max_arr = theta_max * (1-lmda) * np.arange(1,T+1) + theta_max - theta_init

        x_min_arr = np.ones(T) * theta_min - theta_min
        x_max_arr = np.ones(T) * theta_max - theta_min

        # x_min_arr = theta_min / np.pow(lmda,np.arange(0,T)) - lmda * theta_init
        # x_max_arr = theta_max - np.pow(lmda,np.arange(1,T+1))  * theta_init

        # Initialize parent class with constructed parameter arrays
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
        lmda, u_min, u_max, theta_min, theta_max, theta_init = sample.tcl(T)
        return cls(
            T=T,
            lmda=lmda,
            u_min=u_min,
            u_max=u_max,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_init=theta_init,
        )


class TCLn(TCL):
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

    def p(self, A: set) -> float:
        D, e = self.A_b()
        c = np.zeros(self.T)
        c[list(A)] = 1
        u = cp.Variable(self.T)
        peta = cp.Variable()
        constraints = [D @ u <= e, peta <= c.T @ u]
        obj = cp.Maximize(peta)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return peta.value

    def b(self, A: set) -> float:
        D, e = self.A_b()
        c = np.zeros(self.T)
        c[list(A)] = 1
        u = cp.Variable(self.T)
        beta = cp.Variable()
        constraints = [D @ u <= e, c.T @ u <= beta]
        obj = cp.Minimize(beta)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return beta.value


class TCLs(TCL):
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

    def p(self, A: set) -> float:
        D, e = self.A_b()
        c = np.zeros(self.T)
        c[list(A)] = 1
        u = cp.Variable(self.T)
        peta = cp.Variable()
        constraints = [D @ u <= e, c.T @ u <= peta]
        obj = cp.Minimize(peta)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return peta.value

    def b(self, A: set) -> float:
        D, e = self.A_b()
        c = np.zeros(self.T)
        c[list(A)] = 1
        u = cp.Variable(self.T)
        beta = cp.Variable()
        constraints = [D @ u <= e, beta <= c.T @ u]
        obj = cp.Maximize(beta)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return beta.value


class TCL1(TCL):
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

    # def p(self, A: set) -> float:
    #     return len(A) * self.theta_min * (1-self.lmda)

    # def b(self, A: set) -> float:
    #     return len(A) * self.theta_max * (1-self.lmda)

    def b(self, A: set) -> float:
        intervals = split_into_consecutive_ranges(A)

        b_tot = 0
        x = self.theta_init
        t_start = 0

        for I in intervals:
            t = I[0] - t_start
            x = self.x_p(x, t)
            t = I[1] - I[0] + 1
            b_tot += self.b_sufficient(x, t)
            x = self.x_b(x, t)
            t_start = I[1] + 1

        return b_tot

    def p(self, A: set) -> float:
        intervals = split_into_consecutive_ranges(A)

        p_tot = 0
        x = self.theta_init
        t_start = 0

        for I in intervals:
            t = I[0] - t_start
            x = self.x_b(x, t)
            t = I[1] - I[0] + 1
            p_tot += self.p_sufficient(x, t)
            x = self.x_p(x, t)
            t_start = I[1] + 1
        return p_tot

    def b_sub(self, A: set, x_0) -> float:
        x = x_0
        intervals = split_into_consecutive_ranges(A)

        b_tot = 0
        t_start = 0

        for I in intervals:
            t = I[0] - t_start
            x = self.x_p(x, t)
            t = I[1] - I[0] + 1
            b_tot += self.b_interval(x, t)
            x = self.x_b(x, t)
            t_start = I[1] + 1

        return b_tot

    def p_sub(self, A: set, x_0) -> float:
        x = x_0
        intervals = split_into_consecutive_ranges(A)

        p_tot = 0
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

    def v_b_up(self, x_0, t):
        A = set([])
        v_up = np.empty(t)
        for s in np.arange(t):
            A_next = A.union(set([s]))
            v_up[s] = self.b_sub(A_next, x_0) - self.b_sub(A, x_0)
            A = A_next
        return v_up

    def v_b_down(self, x_0, t):
        A = set([])
        v_down = np.empty(t)
        for s in np.arange(t)[::-1]:
            A_next = A.union(set([s]))
            v_down[s] = self.b_sub(A_next, x_0) - self.b_sub(A, x_0)
            A = A_next
        return v_down

    def v_b_up(self, x_0, t):
        A = set([])
        v_up = np.empty(t)
        for s in np.arange(t):
            A_next = A.union(set([s]))
            v_up[s] = self.b_sub(A_next, x_0) - self.b_sub(A, x_0)
            A = A_next
        return v_up

    def v_p_down(self, x_0, t):
        A = set([])
        v_down = np.empty(t)
        for s in np.arange(t)[::-1]:
            A_next = A.union(set([s]))
            v_down[s] = self.p_sub(A_next, x_0) - self.p_sub(A, x_0)
            A = A_next
        return v_down

    def v_p_up(self, x_0, t):
        A = set([])
        v_up = np.empty(t)
        for s in np.arange(t):
            A_next = A.union(set([s]))
            v_up[s] = self.p_sub(A_next, x_0) - self.p_sub(A, x_0)
            A = A_next
        return v_up

    def b_sufficient(self, x_0, t):
        if self.x_b(x_0, t) < self.theta_max:
            return self.u_max * t

        power_dis = x_0 * np.power(self.lmda, np.arange(t + 1))
        power_in = (
            np.cumsum(np.insert(np.power(self.lmda, np.arange(t)), 0, 0)) * self.u_max
        )
        x_t = power_dis + power_in
        t_hits = x_t[1:] < self.theta_max

        if np.all(t_hits):
            return t * self.u_max

        # b_nec = min((m1*t + c1), (m2*t + c2))
        b_nec = self.b_interval(x_0, t)

        v_down = self.v_b_down(x_0, t)
        v_up = self.v_b_up(x_0, t)
        a_down = np.power(self.lmda, -np.arange(t))
        return b_nec - np.dot(a_down, v_down - v_up)

    def p_sufficient(self, x_0, t):
        if self.x_p(x_0, t) > self.theta_min:
            return self.u_min * t

        power_dis = x_0 * np.power(self.lmda, np.arange(t + 1))
        x_t = power_dis
        t_hits = x_t[1:] > self.theta_min

        if np.all(t_hits):
            return t * self.u_min

        p_nec = self.p_interval(x_0, t)
        v_down = self.v_p_down(x_0, t)
        v_up = self.v_p_up(x_0, t)
        a_down = np.power(self.lmda, -np.arange(t))
        return p_nec - np.dot(a_down, v_down - v_up)

    def b_interval(self, x_0, t):
        assert x_0 <= self.theta_max, "Initial x_0 too high"
        power_dis = x_0 * np.power(self.lmda, np.arange(t + 1))
        power_in = (
            np.cumsum(np.insert(np.power(self.lmda, np.arange(t)), 0, 0)) * self.u_max
        )
        x_t = power_dis + power_in
        t_hits = x_t[1:] < self.theta_max

        if np.all(t_hits):
            return t * self.u_max

        t_hit = np.argmax(~t_hits)  # - 1?
        x_t_hit = x_t[t_hit]
        return (
            self.u_max * t_hit
            + self.theta_max
            - self.lmda * x_t_hit
            + (t - t_hit - 1) * self.theta_max * (1 - self.lmda)
        )

    # *np.power(self.lmda, t-1)#TODO put some lmdas in here

    def p_interval(self, x_0, t):
        assert x_0 >= self.theta_min, "Initial x_0 too low"
        power_dis = x_0 * np.power(self.lmda, np.arange(t + 1))
        x_t = power_dis
        t_hits = x_t[1:] > self.theta_min

        if np.all(t_hits):
            return t * self.u_min

        t_hit = np.argmax(~t_hits)  # - 1?
        x_t_hit = x_t[t_hit]
        return (
            self.u_min * t_hit
            + self.theta_min
            - self.lmda * x_t_hit
            + (t - t_hit - 1) * self.theta_min * (1 - self.lmda)
        )


def split_into_consecutive_ranges(input_set):
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


class TCL2(TCL):
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

    def p(self, A: set) -> float:
        return len(A) * self.theta_min * (1 - self.lmda)

    def b(self, A: set) -> float:
        return len(A) * self.theta_max * (1 - self.lmda)

    def b(self, A: set) -> float:
        intervals = split_into_consecutive_ranges(A)

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
        intervals = split_into_consecutive_ranges(A)

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

    def b_interval(self, x_0, t):
        assert x_0 < self.theta_max, "Initial x_0 too high"
        power_dis = x_0 * np.power(self.lmda, np.arange(t + 1))
        power_in = (
            np.cumsum(np.insert(np.power(self.lmda, np.arange(t)), 0, 0)) * self.u_max
        )
        x_t = power_dis + power_in
        t_hits = x_t[1:] < self.theta_max

        if np.all(t_hits):
            return t * self.u_max

        t_hit = np.argmax(~t_hits)  # - 1?
        x_t_hit = x_t[t_hit]

        return (
            self.u_max * t_hit
            + self.theta_max
            - self.lmda * x_t_hit
            + (t - t_hit - 1) * self.theta_max * (1 - self.lmda)
        )

    def p_interval(self, x_0, t):
        assert x_0 > self.theta_min, "Initial x_0 too low"
        power_dis = x_0 * np.power(self.lmda, np.arange(t + 1))
        x_t = power_dis
        t_hits = x_t[1:] > self.theta_min

        if np.all(t_hits):
            return t * self.u_min

        t_hit = np.argmax(~t_hits)  # - 1?
        x_t_hit = x_t[t_hit]

        m0 = self.u_min
        m1 = self.theta_min - self.lmda * x_t_hit
        m2 = self.theta_min * (1 - self.lmda)

        c0 = 0
        c1 = (self.u_min - m1) * t_hit
        c2 = (
            self.u_min * t_hit + self.theta_min - self.lmda * x_t_hit - m2 * (t_hit + 1)
        )

        return max(m0 * t + c0, m1 * t + c1, m2 * t + c2)


def split_into_consecutive_ranges(input_set):
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


class TCL3(TCL):
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
        obj = cp.Minimize(cp.sum(u))
        prob = cp.Problem(obj, constraints)
        return prob.solve()

    def p_interval(self, x_0, t):
        assert x_0 > self.theta_min, "Initial x_0 too low"
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
