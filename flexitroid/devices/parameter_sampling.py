import numpy as np

def der(T):
    # Power bounds (kW)
    u_min = np.full(T, -2.0)  # Can discharge up to 2kW
    u_max = np.full(T, 2.0)  # Can charge up to 2kW

    # Energy bounds (kWh)
    capacity = 10.0  # 10kWh battery
    soc_min = 0.2  # 20% minimum state of charge
    soc_max = 0.8  # 80% maximum state of charge

    x_min = np.full(T, capacity * soc_min)  # 2kWh minimum
    x_max = np.full(T, capacity * soc_max)  # 8kWh maximum
    return u_min, u_max, x_min, x_max

def generation(T):
    # Create sinusoidal generation profile peaking at midday
    t = np.linspace(0, 2 * np.pi, T)
    base_profile = -np.maximum(0, np.random.uniform(0.1) + np.sin(t - np.pi / 2))  # Negative = generation

    # Scale to realistic power bounds (kW)
    rated_power = np.random.uniform(10)  # 5kW rated power
    u_min = rated_power * base_profile
    u_max = np.zeros_like(u_min)  # Can curtail to zero but not consume
    return u_min, u_max


def e1s():
    u_max = 3 + 7*np.random.uniform()
    x_max = 25 + 25*np.random.uniform() 
    x_min = np.random.uniform(x_max)  
    return u_max, x_min, x_max

def v1g(T):
    # Timing parameters
    a = np.random.randint(T-1)
    d = np.random.randint(a+1,T)  # Departure at 7am (or T if shorter horizon)

    # Power and energy parameters
    u_max = np.random.uniform(10) # 
    connected_time = d-a
    e_max = np.random.uniform(connected_time*u_max)  # Maximum 53.3kWh (80% of 66.6kWh battery)
    e_min = np.random.uniform(e_max)  # Minimum 20kWh (30% of 66.6kWh battery)

    return a, d, u_max, e_min, e_max

def v2g(T):
    # Timing parameters
    a = 0  # Arrival at 6pm
    d = min(13, T)  # Departure at 7am (or T if shorter horizon)

    # Power parameters (kW)
    u_min = -11.0  # 11kW discharge
    u_max = 11.0  # 11kW charge

    # Energy parameters (kWh)
    capacity = 100.0  # 100kWh battery
    x_0 = capacity*np.random.uniform(0.2, 0.9)

    x_min = 0.2 * capacity - x_0 # 20% minimum while connected
    x_max = 0.9 * capacity - x_0  # 90% maximum while connected
    e_min = 0.5 * capacity - x_0  # 50% minimum at departure
    e_max = 0.8 * capacity - x_0  # 80% maximum at departure

    return a, d, u_min, u_max, x_min, x_max, e_min, e_max

def e2s():
    # Power parameters (kW)
    u_min = -25.0  # 25kW discharge
    u_max = 25.0  # 25kW charge

    # Energy parameters (kWh)
    capacity = 50.0  # 50kWh storage
    soc_min = 0.1  # 10% minimum
    soc_max = 0.9  # 90% maximum

    x_min = capacity * soc_min  # 5kWh minimum
    x_max = capacity * soc_max  # 45kWh maximum
    return u_min, u_max, x_min, x_max

