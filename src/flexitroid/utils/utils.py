import numpy as np
from scipy.spatial import ConvexHull
import cdd

def get_vertices(A, b):
    """
    Compute the vertices of a polytope defined by Ax <= b using pycddlib.
    
    Parameters:
        A (ndarray): Array of shape (m, n) defining half-space normals.
        b (ndarray): Array of shape (m,) defining right-hand side values.
    
    Returns:
        ndarray: Array of vertices.
    """
    m, n = A.shape
    # Construct H-representation for cddlib: each row is [b_i, -A_i]
    H = np.hstack((b.reshape(m, 1), -A))
    H_list = H.tolist()
    
    # Create a cdd.Matrix and set it to represent inequalities.
    mat = cdd.Matrix(H_list, number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    
    # Create the polyhedron object and get the V-representation.
    poly = cdd.Polyhedron(mat)
    generators = poly.get_generators()
    
    vertices = []
    # The first entry in each row indicates if it's a vertex (1) or a ray (0)
    for row in generators:
        if row[0] == 1:  # Vertex
            vertices.append(list(row[1:]))
    points = np.array(vertices)
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    return np.array(vertices)

def sort_vertices(arr):
    arr = np.unique(arr, axis=0)
    return arr


def project(vertices):
    if len(vertices.shape) == 1:
        n = len(vertices)
    else:
        n = vertices.shape[1]
    R = get_rotation_matrix(n)[:-1]
    return np.dot(vertices, R.T)


def lift(points, b=None):
    single = False
    if len(points.shape) == 1:
        single = True
        points = points.reshape(1, points.shape[0])
    T = points.shape[1] + 1
    R = get_rotation_matrix(T)
    if b is None:
        b = (R @ np.ones(T) * np.mean(np.arange(1, T + 1)))[-1]
    R_inv = np.linalg.inv(R)
    points = np.hstack([points, b * np.ones((points.shape[0], 1))])
    if single:
        return (R_inv @ points.T).T[0]
    return (R_inv @ points.T).T


def visit_edges(p_simplex):
    T = p_simplex.shape[0]
    a = p_simplex[0] * np.ones((T - 1, T - 1))
    a_leaf = np.empty((a.shape[0] + p_simplex[1:].shape[0], a.shape[1]))
    a_leaf[::2, :] = a
    a_leaf[1::2, :] = p_simplex[1:]
    return np.concatenate([a_leaf, p_simplex[1:], p_simplex[1:2]])


def single_minkowski_sum(polytope1, polytope2):
    # polytope1 and polytope2 are lists of 2D or 3D vertices

    # Ensure vertices are in NumPy arrays for easier manipulation
    vertices1 = np.array(polytope1)
    vertices2 = np.array(polytope2)

    # Calculate the Minkowski sum
    sum_vertices = []
    for v1 in vertices1:
        for v2 in vertices2:
            sum_vertices.append(v1 + v2)
    sum_vertices = np.array(sum_vertices)
    return sum_vertices[ConvexHull(sum_vertices).vertices]


def minkowski_sum(flexAssets):
    m_sum = None
    for flexAsset in flexAssets:
        if m_sum is None:
            m_sum = flexAsset.get_vertices()
        else:
            m_sum = single_minkowski_sum(m_sum, flexAsset.get_vertices())
    return m_sum


def get_rotation_matrix(n) -> np.ndarray:
    """Returns rotation matrix that projects points to feasibility set

    Returns:
        np.ndarray: Rotation Matrix
    """
    R = np.identity(n)
    c = np.ones(n)
    for i in range(n - 1):
        c = R @ np.ones(n)
        phi = np.arctan(c[i])
        A = np.identity(n)
        A[i, i] = np.cos(phi)
        A[i, i + 1] = -np.sin(phi)
        A[i + 1, i] = np.sin(phi)
        A[i + 1, i + 1] = np.cos(phi)
        R = A @ R
    return R


def find_normal(points):
    points = np.array(points)
    vectors = points[1:] - points[0]
    _, _, vh = np.linalg.svd(vectors)
    normal = vh[-1]
    return normal


def check_coplanar(simplex):
    plane_points = simplex[:-1]
    n = find_normal(plane_points)
    return len(np.unique((n @ simplex.T).round(decimals=8))) == 1
