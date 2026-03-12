from abc import ABC, abstractmethod
from typing import List, Set, Optional, TypeVar, Generic
import numpy as np
from itertools import permutations
from flexitroid.problems import l_inf, linear, quadratic, signal_tracker
from itertools import combinations


class Flexitroid(ABC):
    """Abstract base class for flexiblity of DERs and aggregations of DERS.

    This class defines the common interface that flexibile entities must implement
    for flexibility set representation and computation.
    """

    @abstractmethod
    def b(self, A: Set[int]) -> float:
        """Compute submodular function b for the g-polymatroid representation."""
        pass

    @abstractmethod
    def p(self, A: Set[int]) -> float:
        """Compute supermodular function p for the g-polymatroid representation."""
        pass

    @property
    @abstractmethod
    def T(self) -> int:
        """Get the time horizon."""
        pass

    def _b_star(self, A: Set) -> float:
        """Extended set function b* for the lifted base polyhedron.

        Args:
            A: A subset of the extended ground set T*.

        Returns:
            Value of b*(A) as defined in the paper.
        """
        if not isinstance(A, set):
            A = set(A)

        if self.T in A:  # t* is in A
            T_set = set(range(self.T))
            return -self.p(T_set - A)
        return self.b(A)

    def greedy(self, c: np.ndarray) -> np.ndarray:
        """Solve a linear program over the g-polymatroid using the greedy algorithm.

        Args:
            c: Cost vector of length T.

        Returns:
            Optimal solution vector.
        """
        # Extend cost vector with c*(t*) = 0
        c_star = np.append(c, 0)

        # Sort indices by non-decreasing cost
        pi = np.argsort(c_star)

        # Initialize solution vector
        v = np.zeros(self.T + 1)

        # Apply greedy algorithm
        S_k = set()
        b_star_prev = 0
        for k in pi:
            S_k.add(int(k))
            b_star = self._b_star(S_k)
            v[k] = b_star - b_star_prev
            b_star_prev = b_star

        # Project solution by removing t* component
        return v[:-1]

    def form_box(self):
        C = np.vstack([np.eye(self.T) + 1, -np.arange(self.T) - 1])
        box = []
        for i, c in enumerate(C):
            box.append(self.greedy(c))
        box = np.array(box)
        return box

    def get_all_vertices(self):
        perms = []
        for t in range(self.T + 1):
            perms.append(list(permutations(np.arange(self.T) + 1 - t)))

        perms = np.array(perms).reshape(-1, self.T)
        V = np.array([self.greedy(c) for c in perms])
        return V

    def solve_l_inf(self, l: np.ndarray = None):
        problem = l_inf.L_inf(self, l)
        problem.solve()
        return problem

    def solve_lp(self, c, A=None, b=None):
        problem = linear.LinearProgram(self, A, b, c)
        problem.solve()
        return problem

    def solve_qp(self, c, Q):
        problem = quadratic.QuadraticProgram(self, Q, c)
        problem.solve()
        return problem
    
    def track_signal(self, signal):
        problem = signal_tracker.SingalTracker(self, signal)
        problem.solve()
        return problem

    def get_g_polymatroid_constraints(self):
        """
        Construct the constraint matrices A and b that define the g-polymatroid Q(p,b).

        The g-polymatroid is defined by constraints:
            p(A) ≤ Σ_{t∈A} u(t) ≤ b(A)  for all subsets A ⊆ [T]

        These are converted to standard form Au ≤ b where:
        - For each subset A, we have:
          * Upper bound: Σ_{t∈A} u(t) ≤ b(A)
          * Lower bound: -Σ_{t∈A} u(t) ≤ -p(A)

        Returns
        -------
        A : np.ndarray
            Constraint matrix of shape (2 * 2^T, T) where each row represents
            a constraint for a subset A ⊆ [T].
        b : np.ndarray
            Right-hand side vector of length 2 * 2^T containing the constraint bounds.

        Raises
        ------
        ValueError
            If T > 8, as the computation becomes infeasible (2^T constraints).

        Notes
        -----
        - This is O(2^T * T) in time and space, so only feasible for moderately small T.
        - The first 2^T rows correspond to upper bounds (b(A) constraints).
        - The last 2^T rows correspond to lower bounds (-p(A) constraints).
        """
        T = self.T
        if T > 8:
            raise ValueError(f"T={T} is too large. This function only supports T ≤ 8 due to exponential complexity (2^T constraints).")
        idxs = list(range(T))
        
        # Total number of constraints: 2 constraints per subset (upper and lower bound)
        num_constraints = 2 * (2 ** T)
        A = np.zeros((num_constraints, T))
        b_vec = np.zeros(num_constraints)
        
        constraint_idx = 0
        
        # Iterate over all subsets A ⊆ [T]
        for size in range(T + 1):
            for combo in combinations(idxs, size):
                A_subset = set(combo)
                
                # Upper bound constraint: Σ_{t∈A} u(t) ≤ b(A)
                # Create row with 1s in positions t∈A
                for t in A_subset:
                    A[constraint_idx, t] = 1.0
                b_vec[constraint_idx] = self.b(A_subset)
                constraint_idx += 1
                
                # Lower bound constraint: -Σ_{t∈A} u(t) ≤ -p(A)
                # Create row with -1s in positions t∈A
                for t in A_subset:
                    A[constraint_idx, t] = -1.0
                b_vec[constraint_idx] = -self.p(A_subset)
                constraint_idx += 1
        
        return A, b_vec

    def in_g_polymatroid_naive(self, u: np.ndarray) -> bool:
        """
        Naively check whether vector u lies in Q(p,b).

        Parameters
        ----------
        u : list or array of length T
            The candidate point in R^T we want to test.
        p : dict or callable
            If dict: p[A] gives p-value for subset A (where A is a frozenset of indices).
            If callable: p(A) returns p-value for subset A.
            Must be supermodular (though we do NOT check supermodularity here).
        b : dict or callable
            If dict: b[A] gives b-value for subset A (where A is a frozenset of indices).
            If callable: b(A) returns b-value for subset A.
            Must be submodular (though we do NOT check submodularity here).

        Returns
        -------
        (bool, A_violation)
            bool = True if u is in Q(p,b), False otherwise
            A_violation = subset A that violates constraints if any, else None

        Notes
        -----
        - This is O(2^T * T) in time, so only feasible for moderately small T.
        - We assume T = len(u).
        - p and b must be provided in a form that can be accessed for each subset A.
        For example, if p is a dict, the key could be frozenset({i1,i2,...}).
        """

        T = len(u)
        idxs = list(range(T))

        # Helper to get p(A) or b(A) no matter if it's a dict or a function
        # Iterate over all subsets
        for size in range(T + 1):
            for combo in combinations(idxs, size):
                A = set(
                    combo
                )  # use a frozenset so it matches dict keys if used that way

                # Sum of u(t) for t in A
                sum_uA = sum(u[t] for t in A)

                # Check lower bound p(A)
                pA = self.p(A)
                if pA is not None:  # assume it must exist
                    if sum_uA < pA:
                        return (False, A)  # found a violation: sum_u(A) < p(A)

                # Check upper bound b(A)
                bA = self.b(A)
                if bA is not None:
                    if sum_uA > bA:
                        return (False, A)  # found a violation: sum_u(A) > b(A)

        # If we never found any violation, then it is in Q(p,b)
        return (True, None)
