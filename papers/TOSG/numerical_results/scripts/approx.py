import numpy as np
from flexitroid.devices.tcl28 import TCLapprox, TCLinner, TCLouter
from flexitroid.utils.population_generator import PopulationGenerator
import matplotlib.pyplot as plt
import csv

def polytope_volume(A, b):
    """
    Compute the volume of a polytope defined by Ax <= b.
    
    Parameters:
    -----------
    A : np.ndarray
        Constraint matrix of shape (m, n)
    b : np.ndarray
        Right-hand side vector of length m
    
    Returns:
    --------
    float
        Volume of the polytope
    """
    from flexitroid.utils.utils import get_vertices
    from scipy.spatial import ConvexHull
    
    try:
        vertices = get_vertices(A, b)
        if len(vertices) == 0:
            return 0.0
        
        # Check if polytope has full dimension
        if vertices.shape[0] < vertices.shape[1] + 1:
            # Polytope is degenerate (lower dimensional)
            return 0.0
        
        hull = ConvexHull(vertices)
        return hull.volume
    except Exception as e:
        # If volume computation fails, return 0
        print(f"Warning: Volume computation failed: {e}")
        return 0.0

def approximation_metric(A, b, A_approx, b_approx):
    """
    Compute a metric comparing two polytopes P = {x | Ax <= b} and 
    P_approx = {x | A_approx x <= b_approx}.
    
    The metric is the volume ratio: vol(P_approx) / vol(P).
    This metric is:
    - 1.0 if the polytopes are identical
    - Less than 1.0 if P_approx is a proper subset of P
    - 0.0 if P_approx is empty or degenerate
    
    Parameters:
    -----------
    A : np.ndarray
        Constraint matrix for exact polytope
    b : np.ndarray
        Right-hand side for exact polytope
    A_approx : np.ndarray
        Constraint matrix for approximate polytope
    b_approx : np.ndarray
        Right-hand side for approximate polytope
    
    Returns:
    --------
    float
        Metric value between 0 and 1
    """
    vol_exact = polytope_volume(A, b)
    vol_approx = polytope_volume(A_approx, b_approx)
    
    if vol_exact == 0:
        # If exact polytope has zero volume, check if they're both empty
        if vol_approx == 0:
            return 1.0  # Both are empty/degenerate, consider them identical
        return 0.0
    
    metric = vol_approx / vol_exact
    return min(1.0, max(0.0, metric))  # Clamp between 0 and 1


def generate_tcl(T, lmda):
    u_min = 0.
    u_max = 1.
    
    theta_min = 1.
    theta_max = 2.
    theta_init = 1.5
    tcl = TCLinner(T, lmda, u_min, u_max, theta_min, theta_max, theta_init)
    return tcl


def get_metric(lmda, T):
    tcl = generate_tcl(T, lmda)

    A, b = tcl.A_b()
    A_approx, b_approx = tcl.get_g_polymatroid_constraints()

    metric_value = approximation_metric(A, b, A_approx, b_approx)
    return metric_value

if __name__ == "__main__":

    T = 5
    print(f"Computing metrics for T={T}")
    lmda_range5 = np.arange(1, 0.66, -0.03)
    lmda_range5 = np.concatenate([lmda_range5, [0.669]])
    metrics5 = [get_metric(lmda, T) for lmda in lmda_range5]

    T = 6
    print(f"Computing metrics for T={T}")
    lmda_range6 = np.arange(1, 0.75, -0.03)

    lmda_range6 = np.concatenate([lmda_range6, [0.7338]])
    metrics6 = [get_metric(lmda, T) for lmda in lmda_range6]

    T = 7
    print(f"Computing metrics for T={T}")
    lmda_range7 = np.arange(1, 0.82, -0.03)

    lmda_range7 = np.concatenate([lmda_range7, [0.82]])
    metrics7 = [get_metric(lmda, T) for lmda in lmda_range7]
    lmda_range7 = np.concatenate([[0.8], lmda_range7])
    metrics7 = np.concatenate([[0.], metrics7])

    with open(f'papers/TOSG/numerical_results/data/approx_metrics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['lmda', 'metric', 'T'])
        for lmda, metric, T in zip(lmda_range7, metrics7, [7]*len(lmda_range7)):
            writer.writerow([lmda, metric, T])
        for lmda, metric, T in zip(lmda_range6, metrics6, [6]*len(lmda_range6)):
            writer.writerow([lmda, metric, T])
        for lmda, metric, T in zip(lmda_range5, metrics5, [5]*len(lmda_range5)):
            writer.writerow([lmda, metric, T])