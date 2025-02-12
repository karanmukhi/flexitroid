import numpy as np
import cvxpy as cp
from flexitroid.flexitroid import Flexitroid

class L_inf():
    def __init__(self, feasible_set: Flexitroid):
        """Initialize the linear program with Dantzig-Wolfe decomposition.
        
        Args:
            X: Flexitroid object representing the feasible set
            A: Constraint matrix
            b: Constraint bounds
            c: Cost vector
        """
        self.feasible_set = feasible_set
        self.epsilon = 1e-6  # Convergence tolerance
        self.max_iter = 1000  # Maximum iterations

        self.lmda = None
        self.v_subset = None
        self.solution = None

    def solve(self):
        """Solve the linear program using Dantzig-Wolfe decomposition.
        
        Returns:
            Optimal solution vector
        """
        if self.lmda == None:
            lmda, v_subset = self.dantzig_wolfe()
            self.lmda = lmda
            self.v_subset = v_subset
            self.solution = lmda@v_subset
    
    def dantzig_wolfe(self):
        v_subset = self.feasible_set.form_box()

        i = 0
        while True:
            i+=1
            print(i, end='\r')
            t = cp.Variable(nonneg=True)
            lmda = cp.Variable(v_subset.shape[0], nonneg=True)

            con_convex = [cp.sum(lmda) == 1]
            con_upper = [lmda@v_subset <= t]
            con_lower = [-lmda@v_subset <= t,]
            constraints = con_convex + con_upper + con_lower

            objective = cp.Minimize(t)
            prob = cp.Problem(objective, constraints)
            prob.solve()

            mu = con_convex[0].dual_value
            pi_plus = con_upper[0].dual_value
            pi_minus = con_lower[0].dual_value
            pi = pi_plus - pi_minus

            new_vertex = self.feasible_set.solve_linear_program(pi)

            reduced_cost = - mu - np.dot(new_vertex, pi)

            if reduced_cost < 1e-9:
                print('Terminating')
                break
            else:
                v_subset = np.vstack([v_subset, new_vertex])

        if i > self.max_iter:
            raise Exception('Did not converge')
        return lmda.value, v_subset
    