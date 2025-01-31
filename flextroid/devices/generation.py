from .general_der import GeneralDER, DERParameters
import numpy as np
from typing import Set


class PV(GeneralDER):
    """Photovoltaic system flexibility set representation.

    This class implements the flexibility set for a PV system with
    curtailment capabilities but no energy storage.
    """

    def __init__(self, u_min: np.ndarray, u_max: np.ndarray):
        """Initialize PV flexibility set with power bounds only.

        Args:
            u_min: Lower bound on power consumption for each timestep.
            u_max: Upper bound on power consumption for each timestep.
        """
        T = len(u_min)
        assert len(u_max) == T, "Power bounds must have same length"
        assert np.all(u_min <= u_max), "Invalid power bounds"

        # Create DER parameters with infinite energy bounds
        params = DERParameters(
            u_min=u_min,
            u_max=u_max,
            x_min=np.full(T, -np.inf),
            x_max=np.full(T, np.inf),
        )
        super().__init__(params)

    def b(self, A: Set[int]) -> float:
        """Compute submodular function b for the g-polymatroid representation.

        Args:
            A: Subset of the ground set T.

        Returns:
            Value of b(A) as defined in Section II-D of the paper
        """
        return np.sum(self.params.u_max[A])

    def p(self, A: Set[int]) -> float:
        """Compute supermodular function p for the g-polymatroid representation.

        Args:
            A: Subset of the ground set T.

        Returns:
            Value of p(A) as defined in Section II-D of the paper
        """
        return np.sum(self.params.u_min[A])
