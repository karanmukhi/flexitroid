from typing import Dict, List
import numpy as np

from flexitroid.devices.general_der import GeneralDER
from flexitroid.devices.pv import PV
from flexitroid.devices.level1 import E1S, V1G
from flexitroid.devices.level2 import E2S, V2G
import flexitroid.utils.device_sampling as sample


class PopulationGenerator:
    """Generator for creating populations of different device types.

    This class handles the generation of multiple device populations with specified
    time horizons and counts, utilizing existing parameter sampling utilities.
    """

    def __init__(
        self,
        T: int,
        pv_count: int = 0,
        der_count: int = 0,
        v1g_count: int = 0,
        e1s_count: int = 0,
        e2s_count: int = 0,
        v2g_count: int = 0,
    ):
        """Initialize the population generator.

        Args:
            T: Length of the time horizon.
        """
        if T <= 0:
            raise ValueError("Time horizon must be positive")
        self.T = T
        self.poppulation = self.generate_population(
            pv_count, der_count, v1g_count, e1s_count, e2s_count, v2g_count
        )

        self.device_list = self.get_all_devices()
        self.N = len(self.device_list)

    def get_all_devices(self) -> List:
        """Get all devices from all populations in a single list.

        Returns:
            List of all devices across all device types.
        """
        all_devices = []
        for devices in self.poppulation.values():
            all_devices.extend(devices)
        return all_devices

    def calculate_indiv_sets(self):
        arr = np.array([device.A_b(False)[1] for device in self.device_list]).T
        large = (
            np.max(arr[np.isfinite(arr)]) * self.T
        )  # Adjust as necessary for your domain
        small = (
            np.min(arr[np.isfinite(arr)]) * self.T
        )  # Adjust as necessary for your domain

        # Replace positive and negative infinities accordingly.
        arr[np.isposinf(arr)] = large
        arr[np.isneginf(arr)] = small
        return arr

    def generate_population(
        self,
        pv_count: int = 0,
        der_count: int = 0,
        v1g_count: int = 0,
        e1s_count: int = 0,
        e2s_count: int = 0,
        v2g_count: int = 0,
    ) -> Dict[str, List]:
        """Generate populations of different device types.

        Args:
            pv_count: Number of PV devices to generate.
            v1g_count: Number of V1G devices to generate.
            e1s_count: Number of E1S devices to generate.
            e2s_count: Number of E2S devices to generate.

        Returns:
            Dictionary containing lists of initialized devices for each type.
        """
        populations = {}

        if pv_count > 0:
            populations["pv"] = [
                PV(self.T, *sample.pv(self.T)) for _ in range(pv_count)
            ]

        if der_count > 0:
            populations["der"] = [
                GeneralDER(sample.der(self.T)) for _ in range(der_count)
            ]

        if v1g_count > 0:
            populations["v1g"] = [
                V1G(self.T, *sample.v1g(self.T)) for _ in range(v1g_count)
            ]

        if e1s_count > 0:
            populations["e1s"] = [
                E1S(self.T, *sample.e1s(self.T)) for _ in range(e1s_count)
            ]

        if e2s_count > 0:
            populations["e2s"] = [
                E2S(self.T, *sample.e2s(self.T)) for _ in range(e2s_count)
            ]

        if v2g_count > 0:
            populations["v2g"] = [
                V2G(self.T, *sample.v2g(self.T)) for _ in range(v2g_count)
            ]

        return populations
