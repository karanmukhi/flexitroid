# p_fast.pyx
import numpy as np
cimport numpy as np

def p_fast(set A, int T, set active,
           np.ndarray[double, ndim=1] u_min,
           np.ndarray[double, ndim=1] u_max,
           np.ndarray[double, ndim=1] x_min,
           np.ndarray[double, ndim=1] x_max):
    cdef set A_c = active - A
    cdef double p = np.sum(u_min[list(A)])
    cdef double b_c = np.sum(u_max[list(A_c)])
    cdef set t_set = set()
    cdef int t
    for t in range(T):
        t_set.add(t)
        # Compute new p as the maximum between current p and the expression below:
        p = max(p,
                x_min[t] - b_c +
                np.sum(u_max[list(A_c - t_set)]) +
                np.sum(u_min[list(A - t_set)]))
        # Update b_c as the minimum between current b_c and the expression below:
        b_c = min(b_c,
                  x_max[t] - p +
                  np.sum(u_min[list(A - t_set)]) +
                  np.sum(u_max[list(A_c - t_set)]))
    return p
