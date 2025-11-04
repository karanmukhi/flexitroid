import numpy as np
import cvxpy as cp


class SingalTracker:
    def __init__(self, feasible_set, signal, max_iters=10000):
        """Initialize the linear program with Dantzig-Wolfe decomposition.

        Args:
            X         : Flexitroid object representing the feasible set
            Q         : (n x n) numpy array (assumed symmetric positive semidefinite)
            c         : (n,) numpy array
            x0        : Initial feasible point in the simplex (n-dimensional numpy array)
            tol       : Tolerance for the duality gap stopping criterion.
            max_iter  : Maximum number of iterations.
        """
        self.signal = signal
        self.T = signal.shape[0]
        self.feasible_set = feasible_set
        self.Q = np.eye(self.T)
        self.c = -signal
        self.max_iter = max_iters  # Maximum iterations
        self.epsilon = 1e-6 * np.linalg.norm(signal) # Convergence tolerance

        self.solution = None
        self.value = None

    def solve(self):
        """Solve the linear program using Dantzig-Wolfe decomposition.

        Returns:
            Optimal solution vector
        """
        if self.solution == None:
            self.solution, self.history = self.frank_wolfe()
            self.value = 0.5 * np.dot(self.solution, self.Q @ self.solution) + np.dot(
                self.c, self.solution
            )
        self.V = self.history["V"]
        self.PI = self.history["c"]
        _, self.lmda = self.converged(self.V)

    def frank_wolfe(self):
        history = {"obj": [], "gap": [], "V": [], 'c': []}

        x = self.feasible_set.greedy(self.c)
        history["V"].append(x)
        history["c"].append(self.c)

        for k in range(self.max_iter):
            # Compute gradient: g = Qx + c
            g = self.Q @ x + self.c

            s = self.feasible_set.greedy(g)

            gap = g.dot(x - s)

            # Record objective and gap
            obj = 0.5 * np.dot(x, self.Q @ x) + np.dot(self.c, x)
            history["obj"].append(obj)
            history["gap"].append(gap)
            history["V"].append(s)
            history["c"].append(g)

            if gap < self.epsilon:
                print("converged")
                break

            d = s - x

            denom = np.dot(d, self.Q @ d)
            if denom > 0:
                gamma = -np.dot(d, self.Q @ x + self.c) / denom
                gamma = np.clip(gamma, 0, 1)
            else:
                # If the quadratic term is zero, we fall back to a step size of 1.
                gamma = 1.0

            # Update x
            x = x + gamma * d
            lp, _ = self.converged(history["V"])
            if lp.value != np.inf:
                print('converged')
                break
           
        return x, history
    
    def converged(self, V):
        V = np.array(V)
        n = V.shape[0]
        lmda = cp.Variable(n, nonneg=True)
        constraints = [lmda@V == self.signal, cp.sum(lmda) == 1]
        prob = cp.Problem(objective=cp.Minimize(1), constraints=constraints)
        prob.solve(solver='GUROBI')
        return prob, lmda.value
