import numpy as np
from flexitroid.devices.generation import PV

# Create a time horizon of 24 hours
T = 24

# Create power bounds that follow a typical daily solar generation pattern
# Assuming peak generation around noon (hour 12)
t = np.arange(T)
peak_power = 5.0  # Maximum power output in kW

# Create a bell-shaped curve for maximum power generation
u_max = peak_power * np.exp(-((t - 12) ** 2) / 20)

# Minimum power is 0 (full curtailment possible)
u_min = np.zeros(T)

# Initialize the PV system
pv = PV(u_min=u_min, u_max=u_max)

# Print the power bounds to verify
print("Hour  |  Min Power  |  Max Power")
print("-" * 35)
for hour in range(T):
    print(f"{hour:4d}  |  {u_min[hour]:9.2f}  |  {u_max[hour]:9.2f}")
