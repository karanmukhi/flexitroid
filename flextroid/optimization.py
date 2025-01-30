"""Core optimization module for g-polymatroid optimization.

This module implements the optimization framework for g-polymatroids,
including the greedy algorithm for solving linear programs and vertex labeling.
"""

from typing import Callable, List, Set, Tuple
import numpy as np


class GPolymatroidOptimizer:
    """Optimizer for g-polymatroid optimization problems.
    
    This class implements optimization over g-polymatroids using the lifting to base
    polyhedron approach and the greedy algorithm for solving linear programs.
    """
    
    def __init__(self, b_func: Callable[[Set], float], p_func: Callable[[Set], float], T: int):
        """Initialize the optimizer.
        
        Args:
            b_func: The submodular function b that defines the g-polymatroid.
            p_func: The supermodular function p that defines the g-polymatroid.
            T: Size of the ground set (time horizon).
        """
        self.b = b_func
        self.p = p_func
        self.T = T
        self._validate_paramodularity()
    
    def _validate_paramodularity(self):
        """Validate that (p,b) is a paramodular pair."""
        # TODO: Implement paramodularity check
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
        S_prev = set()
        for k, idx in enumerate(pi, 1):
            S_k = set(pi[:k])
            v[idx] = self._b_star(S_k) - self._b_star(S_prev)
            S_prev = S_k
        
        # Project solution by removing t* component
        return v[:-1]
    