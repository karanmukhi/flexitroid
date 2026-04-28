import numpy as np
import csv
from pathlib import Path

import datetime
from datetime import datetime

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

C = C.reshape(C.shape[0], -1, 2).mean(axis=2)

T = C.shape[1]

# Configuration
NUM_RUNS = 10
CSV_PATH = DATA_DIR / 'case_study.csv'
POPULATION_CONFIG = {
    'v2g_count': 10,
    'v1g_count': 10,
    'pv_count': 40,
    'e2s_count': 40,
}

# Track whether the file already exists to decide on writing the header
file_exists = CSV_PATH.exists()

# Open file and create writer once
with open(CSV_PATH, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists or CSV_PATH.stat().st_size == 0:
        writer.writerow(["benchmark", "run_id", "t", "value"])

    for run_idx in range(NUM_RUNS):
        # Initialize population and benchmarks for this run
        population = PopulationGenerator(T, **POPULATION_CONFIG)
        run_id = np.random.randint(1000000)
        
        base_profile = population.base_line_consumption()
        g_polymatroid = Aggregator(population)
        general_affine = GeneralAffine(population)
        zonotope = Zonotope(population)

        t = 0
        writer.writerow(['base_line', run_id, t, 0])
        writer.writerow(['g-polymatroid', run_id, t, 0])
        writer.writerow(['general_affine', run_id, t, 0])
        writer.writerow(['zonotope', run_id, t, 0])
        # homothet_projection = HomothetProjection(population)

        # Process each time step
        for t, c in enumerate(C):
            # Progress tracking
            t += 1
            print(f'Run {run_idx+1}/{NUM_RUNS}, Step {t}/{len(C)}             ', end='\r')
            
            # Solve optimization problems
            g_polymatroid_lp = g_polymatroid.greedy(c)
            general_affine.solve_lp(c)
            zonotope.solve_lp(c)

            # Write results for all benchmarks
            writer.writerow(['base_line', run_id, t, c @ base_profile])
            writer.writerow(['g-polymatroid', run_id, t, c @ g_polymatroid_lp])
            writer.writerow(['general_affine', run_id, t, c @ general_affine.lp_x])
            writer.writerow(['zonotope', run_id, t, c @ zonotope.lp_x])
        
        # Flush after each run to ensure data is written
        csvfile.flush()

print(f'\nCompleted {NUM_RUNS} runs. Results saved to {CSV_PATH}')
