# b_fast.pyx
import numpy as np
cimport numpy as np

def b_fast(set A, int T, set active,
           np.ndarray[double, ndim=1] u_min,
           np.ndarray[double, ndim=1] u_max,
           np.ndarray[double, ndim=1] x_min,
           np.ndarray[double, ndim=1] x_max):
    cdef set A_c = active - A
    cdef double b = np.sum(u_max[list(A)])
    cdef double p_c = np.sum(u_min[list(A_c)])
    cdef set t_set = set()
    cdef int t
    for t in range(T):
        t_set.add(t)
        # Update b as the minimum between the current b and the computed expression:
        b = min(b,
                x_max[t] - p_c +
                np.sum(u_min[list(A_c - t_set)]) +
                np.sum(u_max[list(A - t_set)]))
        # Update p_c as the maximum between the current p_c and the computed expression:
        p_c = max(p_c,
                  x_min[t] - b +
                  np.sum(u_max[list(A - t_set)]) +
                  np.sum(u_min[list(A_c - t_set)]))
    return b
