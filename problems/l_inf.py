import cvxpy as cp
import numpy as np

# -------------------------------------------------
# Data for the simple example
# -------------------------------------------------
A = np.array([[1, 2],
              [3, 1]])
b = np.array([5, 6])
c = -np.array([4, 3])

# -------------------------------------------------
# Subproblem / Oracle
# -------------------------------------------------
def subproblem_oracle(pi, sigma):
    """
    Solve the subproblem:
        min_{x in X}  (c^T x) - pi^T (A x) - sigma
    Here, X = { x >= 0 : sum(x) <= 4 } in this toy example.

    Because -sigma is constant w.r.t. x, we really solve:
        min_{x in X}  (c - A^T pi)^T x
    and then adjust the objective by subtracting sigma.
    """
    d = c - np.einsum('ij,i->j', A, pi)  # This is the coefficient for x in the subproblem.

    x_var = cp.Variable(2, nonneg=True)
    constraints = [cp.sum(x_var) <= 4]
    obj = cp.Minimize(d @ x_var)
    prob = cp.Problem(obj, constraints)
    prob.solve()  # or any other LP solver supporting duals

    x_opt = x_var.value
    # The "true" reduced cost includes -sigma:
    reduced_cost = prob.value - sigma
    return x_opt, reduced_cost


# -------------------------------------------------
# Dantzig–Wolfe / Column Generation Loop
# -------------------------------------------------
def dantzig_wolfe_decomposition(A, b, c, max_iters=20, tol=1e-6):
    """
    Implements the column-generation loop (Dantzig–Wolfe).
    Maintains a list of 'columns' = feasible points in X.
    """
    # We need at least one feasible column so that the RMP is not immediately infeasible.
    #
    # Solve the system A x^0 = b to get a feasible x^0 >= 0 (if possible).
    # For this small 2×2 system:
    #   x1 + 2x2 = 5
    #   3x1 + x2 = 6
    # Solve by hand or quickly in Python:
    #   x2 = 9/5 = 1.8, x1 = 1.4
    #
    x0 = np.array([1.4, 1.8])  # This satisfies A x^0 = [5, 6] and is nonnegative.
    columns = [x0]

    iteration = 0
    lam_opt = None
    obj_val = None

    while iteration < max_iters:
        print(iteration)
        k = len(columns)

        # 1) Build the Restricted Master Problem (RMP).
        lam = cp.Variable(k, nonneg=True)

        # Constraint:  A (sum_j lam_j x^j) = b
        # Constraint:  sum_j lam_j = 1
        rmp_constraints = [
            A @ (sum(lam[j]*columns[j] for j in range(k))) == b,
            cp.sum(lam) == 1
        ]

        # Objective:  min sum_j lam_j * (c^T x^j)
        costs = [c @ columns[j] for j in range(k)]
        rmp_obj = cp.Minimize(cp.sum([lam[j]*costs[j] for j in range(k)]))

        rmp_prob = cp.Problem(rmp_obj, rmp_constraints)
        rmp_prob.solve()

        if lam.value is None:
            print("RMP infeasible or solver error.")
            break

        lam_opt = lam.value
        obj_val = rmp_prob.value

        # 2) Get duals from the RMP.  The indexing matches the constraints above.
        dual_pi = rmp_constraints[0].dual_value  # shape = 2, for A x = b
        dual_sigma = rmp_constraints[1].dual_value  # scalar, for sum_j lam_j = 1

        # 3) Subproblem (column-generation / pricing)
        x_new, reduced_cost = subproblem_oracle(dual_pi, dual_sigma)

        print(f"Iteration {iteration}: RMP Obj = {obj_val:.4f}, Reduced cost = {reduced_cost:.4f}")

        # 4) Check if there is an improving column
        if reduced_cost >= -tol:
            print("No more improving columns. Terminating.")
            break
        else:
            columns.append(x_new)

        iteration += 1

    # Recover final solution x* = sum_j lam_j x^j
    x_star = sum(lam_opt[j] * columns[j] for j in range(len(columns)))
    return x_star, obj_val, columns


# -------------------------------------------------
# Run the procedure
# -------------------------------------------------
x_opt, final_obj, cols = dantzig_wolfe_decomposition(A, b, c)
print("\nFinal solution x* =", x_opt)
print("Final objective value =", final_obj)
print("Number of columns used =", len(cols))
