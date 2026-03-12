import numpy as np
import cvxpy as cp


class LinearProgram:
    def __init__(self, feasible_set, A: np.ndarray, b: np.ndarray, c: np.ndarray):
        """Initialize the linear program with Dantzig-Wolfe decomposition.

        Args:
            X: Flexitroid object representing the feasible set
            A: Constraint matrix
            b: Constraint bounds
            c: Cost vector
        """
        self.feasible_set = feasible_set
        T = feasible_set.T
        
        # Ensure A is always 2D: shape (num_constraints, T)
        if A is None or (isinstance(A, np.ndarray) and A.size == 0):
            self.A = np.zeros((0, T))
        else:
            A = np.asarray(A)
            if A.ndim == 1:
                # If A is 1D, reshape to (1, T)
                self.A = A.reshape(1, -1)
            elif A.ndim == 2:
                # If A is already 2D, use it as is
                self.A = A
            else:
                raise ValueError(f"A must be 1D or 2D, got {A.ndim}D")
        
        # Ensure b is always 1D: shape (num_constraints,)
        if b is None or (isinstance(b, np.ndarray) and b.size == 0):
            self.b = np.zeros(0)
        else:
            self.b = np.atleast_1d(b)
        
        self.c = c
        self.epsilon = 1e-6  # Convergence tolerance
        self.max_iter = 1000  # Maximum iterations

        self.lmda = None
        self.v_subset = None
        self.solution = None
        self.value = None

    def solve(self):
        """Solve the linear program using Dantzig-Wolfe decomposition.

        Returns:
            Optimal solution vector
        """
        if self.lmda == None:
            lmda, v_subset = self.dantzig_wolfe()
            self.lmda = lmda
            self.v_subset = v_subset
            self.solution = lmda @ v_subset
            self.value = self.c @ self.solution

    def dantzig_wolfe(self):
        V_subset = self.form_initial_set()
        i = 0

        while i < self.max_iter:
            A_V = np.einsum("ij,kj->ik", self.A, V_subset)
            c_V = np.einsum("j,kj->k", self.c, V_subset)

            y, alpha, lmda = self.solve_dual(A_V, c_V)

            # Handle empty y (no constraints case)
            if y.size == 0:
                d = self.c
            else:
                d = self.c - np.einsum("i,ij->j", y, self.A)
            new_vertex = self.feasible_set.greedy(d)

            if d @ new_vertex - alpha > -self.epsilon:
                break
            V_subset = np.vstack([V_subset, new_vertex])
            i += 1
        if not i < self.max_iter:
            raise Exception("Did not converge")
        return lmda, V_subset

    def form_initial_set(self):
        V_subset = self.feasible_set.form_box()
        while True:
            A_V = np.einsum("ij,kj->ik", self.A, V_subset)

            y, alpha = self.initial_vertex_dual(A_V)

            # Handle empty y (no constraints case)
            if y.size == 0:
                d = np.zeros(self.feasible_set.T)
            else:
                d = -np.einsum("i,ij->j", y, self.A)
            new_vertex = self.feasible_set.greedy(d)

            if d @ new_vertex - alpha > -1e-6:
                break
            V_subset = np.vstack([V_subset, new_vertex])
        return V_subset

    def initial_vertex_dual(self, A_V):
        num_constraints = self.b.shape[0]
        
        if num_constraints == 0:
            # No constraints case: y is empty, so y @ b = 0
            # A_V has shape (0, k), so A_V.T @ y = 0 (k-vector of zeros) when y is empty
            alpha = cp.Variable()
            dual_obj = cp.Maximize(alpha)
            dual_constraints = []
            # A_V.T @ y + alpha <= 0 becomes alpha <= 0 when y is empty
            dual_constraints.append(alpha <= 0)
            dual_constraints.append(alpha <= 1)
            dual_constraints.append(-alpha <= 1)
            y_value = np.zeros(0)
        else:
            y = cp.Variable(num_constraints, neg=True)
            alpha = cp.Variable()
            dual_obj = cp.Maximize(y @ self.b + alpha)
            dual_constraints = []
            dual_constraints.append(A_V.T @ y + alpha <= 0)
            dual_constraints.append(alpha <= 1)
            dual_constraints.append(-alpha <= 1)
            y_value = y.value

        dual_prob = cp.Problem(dual_obj, dual_constraints)
        dual_prob.solve(solver=cp.GUROBI)

        return y_value, alpha.value

    def solve_dual(self, A_V, c_V):
        num_constraints = self.b.shape[0]
        
        if num_constraints == 0:
            # No constraints case: y is empty, so y @ b = 0
            alpha = cp.Variable()
            dual_obj = cp.Maximize(alpha)
            # A_V.T @ y = 0 when y is empty, so constraint becomes alpha <= c_V
            dual_constraints = [alpha <= c_V]
            y_value = np.zeros(0)
        else:
            y = cp.Variable(num_constraints, neg=True)
            alpha = cp.Variable()
            dual_obj = cp.Maximize(y @ self.b + alpha)
            dual_constraints = [A_V.T @ y + alpha <= c_V]
            y_value = y.value

        dual_prob = cp.Problem(dual_obj, dual_constraints)
        dual_prob.solve(solver=cp.GUROBI)

        return y_value, alpha.value, dual_constraints[0].dual_value
