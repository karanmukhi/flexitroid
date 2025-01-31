from .general_der import GeneralDER, DERParameters
import numpy as np
from typing import Set

class V1G(GeneralDER):
    """Vehicle-to-Grid Level 1 flexibility set representation.
    
    This class implements the flexibility set for an EV with
    unidirectional charging capabilities (charging only, no discharging)
    and battery storage.
    
    The V1G has time-dependent constraints based on arrival and departure times:
    - Power bounds are 0 outside the charging window (before arrival, after departure)
    - State of charge has different bounds before and after departure
    - Only allows positive power flow (charging only)
    """
    def __init__(self, T: int, a: int, d: int, 
                 u_max: float,
                 e_min: float, e_max: float):
        """Initialize V1G flexibility set with time-dependent constraints.
        
        Args:
            T: Time horizon length
            a: Arrival time
            d: Departure time
            u_max: Maximum power consumption during charging window
            x_min: Minimum state of charge before departure
            x_max: Maximum state of charge before departure
            e_min: Minimum state of charge after departure
            e_max: Maximum state of charge after departure
        """
        assert 0 <= a < d <= T, "Invalid arrival/departure times"
        assert 0 <= u_max, "Invalid power bounds (must be non-negative)"
        assert e_min <= e_max, "Invalid post-departure SoC bounds"
        
        # Create power bound arrays with zeros outside charging window
        u_min_arr = np.zeros(T)
        u_max_arr = np.zeros(T)
        u_max_arr[a:d] = u_max
        
        # Create SoC bound arrays with different constraints before/after departure
        x_min_arr = np.full(T, -np.inf)  # Initialize with no constraints
        x_max_arr = np.full(T, np.inf)   # Initialize with no constraints
        
        # Set SoC bounds after departure
        x_min_arr[d-1:] = e_min
        x_max_arr[d-1:] = e_max
        
        # Initialize parent class with constructed parameter arrays
        params = DERParameters(
            u_min=u_min_arr,
            u_max=u_max_arr,
            x_min=x_min_arr,
            x_max=x_max_arr
        )
        self.major = self._get_major(u_max, e_max, a, d)
        self.minor = self._get_major(u_max, e_min, a, d)[::-1]
        self.active = set(range(a,d))
        super().__init__(params)

    def _get_major(self, u_max: float, e_max: float, a: int, d: int) -> np.ndarray:
        major = np.zeros(shape=(d-a))
        full_power_time = int(e_max // u_max)
        major[:full_power_time] = u_max
        major[full_power_time] = e_max % u_max
        return major
    

    def b(self, A: Set[int]) -> float:
        """Compute submodular function b for the g-polymatroid representation.
        
        Args:
            A: Subset of the ground set T.
            
        Returns:
            Value of b(A) as defined in Section II-D of the paper
        """
        on_times = self.active.intersection(A)
        on_time = len(on_times)
        return np.sum(self.major[:on_time])
        
    def p(self, A: Set[int]) -> float:
        """Compute supermodular function p for the g-polymatroid representation.
        
        Args:
            A: Subset of the ground set T.
            
        Returns:
            Value of p(A) as defined in Section II-D of the paper
        """
        on_times = self.active.intersection(A)
        on_time = len(on_times)
        return np.sum(self.minor[:on_time])



class E1S(V1G):
    """Energy Storage System Level 1 flexibility set representation.
    
    This class implements the flexibility set for a stationary
    energy storage system with unidirectional power flow (charging only).
    """
    def __init__(self, u_max: float, x_min: float, x_max: float, T: int):
        """Initialize E1S flexibility set with constant power and energy bounds.
        
        Args:
            u_max: Maximum power consumption (constant over time).
            x_min: Lower bound on state of charge (constant over time).
            x_max: Upper bound on state of charge (constant over time).
            T: Time horizon length.
        """
        # Call V1G constructor with:
        # - arrival time = 0 (available from start)
        # - departure time = T (available until end)
        # - same final SoC bounds as continuous bounds
        super().__init__(T=T, a=0, d=T, u_max=u_max, e_min=x_min, e_max=x_max)

