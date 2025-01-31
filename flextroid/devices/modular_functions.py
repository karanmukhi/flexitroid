"""Module for super- and submodular function implementations.

This module provides shared functionality for computing submodular (b) and
supermodular (p) functions used in the g-polymatroid representation of
DER flexibility sets.
"""

from typing import Set
import numpy as np


def compute_b_general(A: Set[int], u_min: np.ndarray, u_max: np.ndarray,
                     x_min: np.ndarray, x_max: np.ndarray) -> float:
    if not A:
        return 0.0
        
    t_max = max(A)
    T_t = get_time_interval(t_max)
    A_complement = T_t - A
    b = sum_over_subset(A, u_max)
    p_c = sum_over_subset(A_complement, u_min)
    t_set = set()
    
    for t in range(t_max):
        t_set.add(t)
        b = min(
            b,
            x_max[t]
            - p_c
            + sum_over_subset(A_complement - t_set, u_min)
            + sum_over_subset(A - t_set, u_max)
        )
        p_c = max(
            p_c,
            x_min[t]
            - b
            + sum_over_subset(A - t_set, u_max)
            + sum_over_subset(A_complement - t_set, u_min)
        )
    return b

def compute_p_general(A: Set[int], u_min: np.ndarray, u_max: np.ndarray,
                     x_min: np.ndarray, x_max: np.ndarray) -> float:
    if not A:
        return 0.0
        
    t_max = max(A)
    T_t = get_time_interval(t_max)
    A_complement = T_t - A
    p = sum_over_subset(A, u_min)
    b_c = sum_over_subset(A_complement, u_max)
    t_set = set()
    
    for t in range(t_max):
        t_set.add(t)
        p = max(
            p,
            x_min[t]
            - b_c
            + sum_over_subset(A_complement - t_set, u_max)
            + sum_over_subset(A - t_set, u_min)
        )
        b_c = min(
            b_c,
            x_max[t]
            - p
            + sum_over_subset(A - t_set, u_min)
            + sum_over_subset(A_complement - t_set, u_max)
        )
    return p

def compute_b_pv(A: Set[int], u_max: np.ndarray) -> float:
    return sum_over_subset(A, u_max)

def compute_p_pv(A: Set[int], u_min: np.ndarray) -> float:
    return sum_over_subset(A, u_min)

def compute_b_ess(A: Set[int], major: np.ndarray) -> float:
    ...

def compute_p_ess(A: Set[int], major: np.ndarray) -> float:
    ...

def compute_b_v1g(A: Set[int], u_max: np.ndarray) -> float:
    ...

def compute_p_v1g(A: Set[int], u_min: np.ndarray) -> float:
    ...
