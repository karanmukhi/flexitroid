#!/usr/bin/env python3
"""
Test script for flexitroid and benchmarks.
This script sets up the necessary paths for importing flexitroid and benchmarks modules.
"""

import sys
from pathlib import Path

# Add current directory to path for flexitroid and benchmarks imports
# Now we can import flexitroid and benchmarks
import numpy as np

# Flexitroid imports
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.population_generator import PopulationGenerator
from flexitroid.utils.cost import generate_energy_price_curve
from flexitroid.flexitroid import Flexitroid

# Benchmark imports
from benchmarks.general_affine import GeneralAffine
from benchmarks.zonotope import Zonotope
from benchmarks.homothet import HomothetProjection
from benchmarks.benchmark import Benchmark, InnerApproximation

# Example usage/test code goes here
if __name__ == "__main__":
    print("Testing flexitroid and benchmarks imports...")
    
    # Example: Create a simple test
    T = 24
    print(f"Creating population with T={T}")
    
    try:
        population = PopulationGenerator(
            T,
            v2g_count=5,
            v1g_count=5,
            pv_count=10,
            e2s_count=10,
        )
        print(f"✓ PopulationGenerator created successfully")
        print(f"  - N: {population.N}")
        print(f"  - T: {population.T}")
        
        # Test Aggregator
        print("\nTesting Aggregator...")
        aggregator = Aggregator(population)
        print("✓ Aggregator created successfully")
        
        # Test benchmarks
        print("\nTesting benchmarks...")
        general_affine = GeneralAffine(population)
        print("✓ GeneralAffine created successfully")
        
        zonotope = Zonotope(population)
        print("✓ Zonotope created successfully")
        
        homothet = HomothetProjection(population)
        print("✓ HomothetProjection created successfully")
        
        # Test cost function
        print("\nTesting cost function...")
        c = generate_energy_price_curve(T)
        print(f"✓ Cost curve generated: shape={c.shape}")
        
        print("\n✓ All imports and basic functionality working!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

