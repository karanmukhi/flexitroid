import numpy as np


def generate_energy_price_curve(T, noise_std=5):
    """
    Generate a simplified energy price curve over one day with added noise.

    Parameters:
      T (int): Number of time steps over 24 hours.
      noise_std (float): Standard deviation of the Gaussian noise.

    Returns:
      time (numpy array): Array of time points (in hours).
      price (numpy array): Energy prices at each time step with added noise.
    """
    # Create a time vector from 0 to 24 hours
    time = np.linspace(0, 24, T)

    # Define a base price (this is arbitrary; real prices are far more complex)
    base_price = 50  # baseline price in $/MWh

    # Model a morning peak (e.g., around 8 AM)
    morning_peak = 20 * np.exp(-((time - 8) ** 2) / (2 * 2**2))

    # Model an evening peak (e.g., around 6 PM)
    evening_peak = 30 * np.exp(-((time - 18) ** 2) / (2 * 2**2))

    # Combine the baseline with the two peaks
    price = base_price + morning_peak + evening_peak

    # Add Gaussian noise to simulate random fluctuations
    noise = np.random.normal(loc=0, scale=noise_std, size=T)
    price_with_noise = price + noise

    return price_with_noise
