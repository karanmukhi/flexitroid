from .general_der import GeneralDER, DERParameters
import numpy as np
import flexitroid.utils.device_sampling as sample
import cvxpy as cp
from .tcl import TCL

class TCLzone(TCL):
    def __init__(self, s, T, lmda, u_min, u_max, theta_min, theta_max, theta_init):
        assert s <= T, "Invalid value for s"
        super().__init__(
            T=T,
            lmda=lmda,
            u_min=u_min,
            u_max=u_max,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_init=theta_init,
        )
        self.s = s

    def get_generator(self):
        I = np.eye(self.T)
        r = np.zeros(self.T) 
        r[:self.s] = 1
        J = np.vstack([np.roll(r, i) for i in range(self.T-self.s+1)]).T
        return np.hstack([I, J])
