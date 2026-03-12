import pytest
import numpy as np
from flexitroid.flexitroid import Flexitroid


class MockFlexitroid(Flexitroid):
    """Mock implementation of Flexitroid for testing."""

    def __init__(self, time_horizon: int):
        self._T = time_horizon

    @property
    def T(self) -> int:
        return self._T

    def b(self, A: set) -> float:
        # Simple submodular function for testing
        return len(A)

    def p(self, A: set) -> float:
        # Simple supermodular function for testing
        return 2 * len(A)


def test_b_star_computation():
    """Test the _b_star method computation."""
    flex = MockFlexitroid(time_horizon=3)

    # Test when t* is not in A
    A = {0, 1}
    assert flex._b_star(A) == 2  # Should equal b(A)

    # Test when t* is in A
    A_with_t_star = {0, 1, 3}  # 3 is t* since T=3
    T_set = {0, 1, 2}
    expected = -flex.p(T_set - A_with_t_star)
    assert flex._b_star(A_with_t_star) == expected


def test_solve_linear_program():
    print("dd")
    """Test the linear program solver with a simple cost vector."""
    flex = MockFlexitroid(time_horizon=3)

    # Test with a simple cost vector
    c = np.array([1.0, 2.0, 3.0])
    solution = flex.greedy(c)

    # Basic checks
    assert len(solution) == 3
    assert isinstance(solution, np.ndarray)
    assert all(isinstance(x, (int, float)) for x in solution)
