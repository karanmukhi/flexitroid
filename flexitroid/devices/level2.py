from .general_der import GeneralDER, DERParameters
import numpy as np
import flexitroid.utils.device_sampling as sample


class V2G(GeneralDER):
    """Vehicle-to-Grid Level 2 flexibility set representation.

    This class implements the flexibility set for an EV with
    bidirectional charging capabilities (V2G) and battery storage.

    The V2G has time-dependent constraints based on arrival and departure times:
    - Power bounds are 0 outside the charging window (before arrival, after departure)
    - State of charge has different bounds before and after departure, to reflect consumer requirements when the vehicle departs
    - Supports bidirectional power flow (both charging and discharging)
    """

    def __init__(
        self,
        T: int,
        a: int,
        d: int,
        u_min: float,
        u_max: float,
        x_min: float,
        x_max: float,
        e_min: float,
        e_max: float,
    ):
        """Initialize EV flexibility set with time-dependent constraints.

        Args:
            T: Time horizon length
            a: Arrival time
            d: Departure time
            u_min: Minimum power consumption during charging window
            u_max: Maximum power consumption during charging window
            x_min: Minimum state of charge before departure
            x_max: Maximum state of charge before departure
            e_min: Minimum state of charge after departure
            e_max: Maximum state of charge after departure
        """
        assert 0 <= a < d <= T, "Invalid arrival/departure times"
        assert u_min <= u_max, "Invalid power bounds"
        assert x_min <= x_max, "Invalid pre-departure SoC bounds"
        assert e_min <= e_max, "Invalid post-departure SoC bounds"

        # Create power bound arrays with zeros outside charging window
        u_min_arr = np.zeros(T)
        u_max_arr = np.zeros(T)
        u_min_arr[a:d] = u_min
        u_max_arr[a:d] = u_max

        # Create SoC bound arrays with different constraints before/after departure
        x_min_arr = np.full(T, np.sum(u_min_arr))  # Initialize with no constraints
        x_max_arr = np.full(T, np.sum(u_max_arr))  # Initialize with no constraints

        # Set SoC bounds before departure
        x_min_arr[:d] = x_min
        x_max_arr[:d] = x_max

        # Set SoC bounds after departure
        x_min_arr[d:] = e_min
        x_max_arr[d:] = e_max

        # Initialize parent class with constructed parameter arrays
        params = DERParameters(
            u_min=u_min_arr, u_max=u_max_arr, x_min=x_min_arr, x_max=x_max_arr
        )
        super().__init__(params)
        self.a = a
        self.d = d
        self.u_max = u_max
        self.u_min = u_min
        self.x_min = x_min
        self.x_max = x_max
        self.e_min = e_min
        self.e_max = e_max
        self.active = set(range(a, d))

    @classmethod
    def example(cls, T: int = 24) -> "V2G":
        """Create an example V2G (bidirectional EV) with typical parameters.

        Creates an EV that:
        - Arrives at 6pm (hour 18)
        - Departs at 7am next day (hour 31)
        - Can charge at 11kW and discharge at 11kW (typical V2G)
        - Maintains 20-90% SoC while connected
        - Requires 50-80% SoC at departure

        Args:
            T: Number of timesteps (default 24 for hourly resolution)

        Returns:
            V2G instance with example parameters
        """
        a, d, u_min, u_max, x_min, x_max, e_min, e_max = sample.v2g(T)
        return cls(
            T=T,
            a=a,
            d=d,
            u_min=u_min,
            u_max=u_max,
            x_min=x_min,
            x_max=x_max,
            e_min=e_min,
            e_max=e_max,
        )


class E2S(GeneralDER):
    """Energy storage system flexibility set representation.

    This class implements the flexibility set for a stationary
    energy storage system with bidirectional power flow.
    """

    def __init__(self, T: int, u_min: float, u_max: float, x_min: float, x_max: float):
        """Initialize ESS flexibility set with constant power and energy bounds.

        Args:
            u_min: Lower bound on power consumption (constant over time).
            u_max: Upper bound on power consumption (constant over time).
            x_min: Lower bound on state of charge (constant over time).
            x_max: Upper bound on state of charge (constant over time).
            T: Time horizon length.
        """
        assert u_min <= u_max, "Invalid power bounds"
        assert x_min <= x_max, "Invalid energy bounds"

        # Create DER parameters with constant bounds
        params = DERParameters(
            u_min=np.full(T, u_min),
            u_max=np.full(T, u_max),
            x_min=np.full(T, x_min),
            x_max=np.full(T, x_max),
        )
        super().__init__(params)

    @classmethod
    def example(cls, T: int = 24) -> "E2S":
        """Create an example E2S (bidirectional storage) with typical parameters.

        Creates a storage system that:
        - Has 50kWh capacity
        - Can charge/discharge at 25kW
        - Maintains 10-90% state of charge

        Args:
            T: Number of timesteps (default 24 for hourly resolution)

        Returns:
            E2S instance with example parameters
        """
        # Power parameters (kW)
        u_min = -25.0  # 25kW discharge
        u_max = 25.0  # 25kW charge

        # Energy parameters (kWh)
        capacity = 50.0  # 50kWh storage
        soc_min = 0.1  # 10% minimum
        soc_max = 0.9  # 90% maximum

        x_min = capacity * soc_min  # 5kWh minimum
        x_max = capacity * soc_max  # 45kWh maximum

        u_min, u_max, x_min, x_max = sample.e2s(T)
        return cls(T=T, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)
