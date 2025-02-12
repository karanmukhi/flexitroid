import numpy as np
import cvxpy as cp
from flexitroid.flexitroid import Flexitroid

class FrankWolfe():
    def __init__(self, feasible_set: Flexitroid, A: np.ndarray, b: np.ndarray, c: np.ndarray):
        """Initialize the linear program with Dantzig-Wolfe decomposition.
        
        Args:
            X: Flexitroid object representing the feasible set
            A: Constraint matrix
            b: Constraint bounds
            c: Cost vector
        """
        self.feasible_set = feasible_set
        self.A = A
        self.b = b
        self.c = c
        self.epsilon = 1e-6  # Convergence tolerance
        self.max_iter = 1000  # Maximum iterations

    def solve(self):
        """Solve the linear program using Dantzig-Wolfe decomposition.
        
        Returns:
            Optimal solution vector
        """
        lmda, v_subset = self.dantzig_wolfe()
        self.lmda = lmda
        self.v_subset = v_subset
        self.solution = lmda@v_subset
    
    def dantzig_wolfe(self):
        V_subset = self.form_initial_set()
        i=0

        while i<self.max_iter:
            print(i, end='\r')
            A_V = np.einsum('ij,kj->ik', self.A, V_subset)
            c_V = np.einsum('j,kj->k', self.c, V_subset)


            y, alpha, lmda = self.solve_dual(A_V, c_V)

            d = self.c - np.einsum('i,ij->j', y, self.A)
            new_vertex = self.feasible_set.solve_linear_program(d)

            if d@new_vertex - alpha > -self.epsilon:
                break
            V_subset = np.vstack([V_subset, new_vertex])
            i+=1
        if not i < self.max_iter:
            raise Exception('Did not converge')
        return lmda, V_subset
    
    def form_initial_set(self):
        V_subset = self.feasible_set.form_box()
        while True:
            A_V = np.einsum('ij,kj->ik', self.A, V_subset)

            y, alpha = self.initial_vertex_dual(A_V)

            d = - np.einsum('i,ij->j', y, self.A)
            new_vertex = self.feasible_set.solve_linear_program(d)

            if d@new_vertex - alpha > -1e-6:
                break
            V_subset = np.vstack([V_subset, new_vertex])
        return V_subset

    
    def initial_vertex_dual(self, A_V):
        y = cp.Variable(self.b.shape[0], neg=True)
        alpha = cp.Variable()

        dual_obj = cp.Maximize(y@self.b + alpha)

        dual_constraints = []
        dual_constraints.append(A_V.T @ y + alpha <= 0)
        dual_constraints.append(alpha <= 1)
        dual_constraints.append(-alpha <= 1)

        dual_prob = cp.Problem(dual_obj, dual_constraints)
        dual_prob.solve()

        return y.value, alpha.value
    
    def solve_dual(self, A_V, c_V):
    
        y = cp.Variable(self.b.shape[0], neg=True)
        alpha = cp.Variable()

        dual_obj = cp.Maximize(y@self.b + alpha)
        dual_constraints = [A_V.T @ y + alpha <= c_V]
        dual_prob = cp.Problem(dual_obj, dual_constraints)
        dual_prob.solve()

        return y.value, alpha.value, dual_constraints[0].dual_value
