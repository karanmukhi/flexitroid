from abc import ABC, abstractmethod
from typing import List, Set, Optional, TypeVar, Generic
import numpy as np
from itertools import permutations
from flexitroid.problems import l_inf, linear, quadratic



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

        T_set = set(range(self.T))
        if self.T in A:  # t* is in A
            return -self.p(T_set - A)
        return self.b(A)

    def solve_linear_program(self, c: np.ndarray) -> np.ndarray:
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
            box.append(self.solve_linear_program(c))
        box = np.array(box)
        return box

    def get_all_vertices(self):
        perms = []
        for t in range(self.T + 1):
            perms.append(list(permutations(np.arange(self.T) + 1 - t)))

        perms = np.array(perms).reshape(-1, self.T)
        V = np.array([self.solve_linear_program(c) for c in perms])
        return V

    def solve_l_inf(self, l:np.ndarray = None):
        problem = l_inf.L_inf(self, l)
        problem.solve()
        return problem
    
    def solve_lp(self, c):
        problem = linear.LinearProgram(self, c)
        problem.solve()
        return problem
    
    def solve_qp(self, c, Q):
        problem = quadratic.QuadraticProgram(self, Q, c)
        problem.solve()
        return problem
