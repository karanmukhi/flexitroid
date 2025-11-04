import numpy as np
import cvxpy as cp


class QuadraticProgram:
    def __init__(self, feasible_set, Q, c, max_iters=1000):
        """Initialize the linear program with Dantzig-Wolfe decomposition.

        Args:
            X         : Flexitroid object representing the feasible set
            Q         : (n x n) numpy array (assumed symmetric positive semidefinite)
            c         : (n,) numpy array
            x0        : Initial feasible point in the simplex (n-dimensional numpy array)
            tol       : Tolerance for the duality gap stopping criterion.
            max_iter  : Maximum number of iterations.
        """
        self.feasible_set = feasible_set
        self.Q = Q
        self.c = c
        self.max_iter = max_iters  # Maximum iterations
        self.epsilon = 1e-3 * np.linalg.norm(c) # Convergence tolerance

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

    def frank_wolfe(self):
        history = {"obj": [], "gap": [], "V": []}

        x = self.feasible_set.greedy(self.c)
        history["V"].append(self.c)

        for k in range(self.max_iter):
            # Compute gradient: g = Qx + c
            g = self.Q @ x + self.c

            s = self.feasible_set.greedy(g)

            gap = g.dot(x - s)

            # Record objective and gap
            obj = 0.5 * np.dot(x, self.Q @ x) + np.dot(self.c, x)
            history["obj"].append(obj)
            history["gap"].append(gap)
            history["V"].append(g)

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
        return x, history
