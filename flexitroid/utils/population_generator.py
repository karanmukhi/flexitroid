from typing import Dict, List
import numpy as np

from flexitroid.devices.general_der import GeneralDER
from flexitroid.devices.pv import PV
from flexitroid.devices.level1 import E1S, V1G
from flexitroid.devices.level2 import E2S, V2G
from flexitroid.devices.tcl28 import TCLinner
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
        tcl_count: int = 0,
    ):
        """Initialize the population generator.

        Args:
            T: Length of the time horizon.
        """
        if T <= 0:
            raise ValueError("Time horizon must be positive")
        self.T = T
        self.device_groups = self.generate_population(
            pv_count, der_count, v1g_count, e1s_count, e2s_count, v2g_count, tcl_count
        )

        self.device_list = self.get_all_devices()
        self.N = len(self.device_list)
        self.device_types = ["pv", "der", "v1g", "e1s", "e2s", "v2g", "tcl"]

    def get_all_devices(self) -> List:
        """Get all devices from all populations in a single list.

        Returns:
            List of all devices across all device types.
        """
        all_devices = []
        for devices in self.device_groups.values():
            all_devices.extend(devices)
        return all_devices

    def calculate_indiv_bs(self):
        arr = np.array([device.A_b()[1] for device in self.device_list]).T
        return arr

    def calculate_indiv_As(self):
        arr = np.array([device.A_b()[0] for device in self.device_list]).T
        return arr

    def base_line_consumption(self):
        """Aggregates the baseline consumption of all devices in the population.
        Calculated by assuming households (PV + ESS) minimize external consumption.
        And EVSEs (V1G + ESS) charge asap
        """
        assert len(self.device_groups["pv"]) == len(self.device_groups["e2s"])
        assert len(self.device_groups["der"]) == 0
        assert len(self.device_groups["e1s"]) == 0

        A = []
        for i in range(len(self.device_groups["pv"])):
            l = self.device_groups["pv"][i].params.u_min
            A.append(self.device_groups["e2s"][i].solve_l_inf(l).solution)
        c1g = np.arange(self.T)[::-1]

        for v1g in self.device_groups["v1g"]:
            A.append(v1g.greedy(c1g))

        for v2g in self.device_groups["v2g"]:
            A.append(
                V1G(
                    self.T, v2g.a, v2g.d, v2g.u_max, v2g.e_min, v2g.e_max
                ).greedy(c1g)
            )
        return np.sum(A, axis=0)

    def generate_population(
        self,
        pv_count: int = 0,
        der_count: int = 0,
        v1g_count: int = 0,
        e1s_count: int = 0,
        e2s_count: int = 0,
        v2g_count: int = 0,
        tcl_count: int = 0,
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
        else:
            populations["pv"] = []

        if der_count > 0:
            populations["der"] = [
                GeneralDER(sample.der(self.T)) for _ in range(der_count)
            ]
        else:
            populations["der"] = []

        if v1g_count > 0:
            populations["v1g"] = [
                V1G(self.T, *sample.v1g(self.T)) for _ in range(v1g_count)
            ]
        else:
            populations["v1g"] = []

        if e1s_count > 0:
            populations["e1s"] = [
                E1S(self.T, *sample.e1s(self.T)) for _ in range(e1s_count)
            ]
        else:
            populations["e1s"] = []

        if e2s_count > 0:
            populations["e2s"] = [
                E2S(self.T, *sample.e2s(self.T)) for _ in range(e2s_count)
            ]
        else:
            populations["e2s"] = []

        if v2g_count > 0:
            populations["v2g"] = [
                V2G(self.T, *sample.v2g(self.T)) for _ in range(v2g_count)
            ]
        else:
            populations["v2g"] = []

        if tcl_count > 0:
            populations["tcl"] = [
                TCLinner(self.T, *sample.tcl(self.T)) for _ in range(tcl_count)
            ]
        else:
            populations["tcl"] = []

        return populations
