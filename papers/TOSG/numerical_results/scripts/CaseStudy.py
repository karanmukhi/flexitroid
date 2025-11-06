import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from pathlib import Path

import requests
import datetime
from datetime import datetime, timedelta

import flexitroid.utils.elexon_api as elexon_api
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.population_generator import PopulationGenerator
from benchmarks.general_affine import GeneralAffine
from benchmarks.zonotope import Zonotope
from benchmarks.homothet import HomothetProjection
from flexitroid.utils.cost import generate_energy_price_curve


# Get script directory and set up data directory path
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Save day ahead price curves to data directory
filename = DATA_DIR / f"day_ahead_price_curves_{datetime(2024, 11, 1).strftime('%Y-%m-%dT00:00Z')}_to_{datetime(2024, 12, 1).strftime('%Y-%m-%dT23:30Z')}.csv"

elexon_api.cache_day_ahead_price_curves(datetime(2024, 11, 1), datetime(2024, 12, 1), filename=str(filename))
C = np.genfromtxt(filename, delimiter=",")


T = 4
C = C[:,:T]

# Configuration
NUM_RUNS = 3
CSV_PATH = DATA_DIR / 'case_study.csv'
POPULATION_CONFIG = {
    'v2g_count': 10,
    'v1g_count': 10,
    'pv_count': 40,
    'e2s_count': 40,
}

# Delete existing file if it exists
if CSV_PATH.exists():
    CSV_PATH.unlink()

# Open file and create writer once
with open(CSV_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    for run_id in range(NUM_RUNS):
        # Initialize population and benchmarks for this run
        population = PopulationGenerator(T, **POPULATION_CONFIG)
        
        base_profile = population.base_line_consumption()
        g_polymatroid = Aggregator(population)
        general_affine = GeneralAffine(population)
        homothet_projection = HomothetProjection(population)

        # Process each time step
        for t, c in enumerate(C):
            # Progress tracking
            print(f'Run {run_id+1}/{NUM_RUNS}, Step {t+1}/{len(C)}', end='\r')
            
            # Solve optimization problems
            g_polymatroid_lp = g_polymatroid.greedy(c)
            general_affine.solve_lp(c)
            homothet_projection.solve_lp(c)

            # Write results for all benchmarks
            writer.writerow(['base_line', run_id, t, c @ base_profile])
            writer.writerow(['g-polymatroid', run_id, t, c @ g_polymatroid_lp])
            writer.writerow(['general_affine', run_id, t, c @ general_affine.lp_x])
            writer.writerow(['homothet', run_id, t, c @ homothet_projection.lp_x])
        
        # Flush after each run to ensure data is written
        csvfile.flush()

print(f'\nCompleted {NUM_RUNS} runs. Results saved to {CSV_PATH}')
