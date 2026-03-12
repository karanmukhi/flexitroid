import numpy as np
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.population_generator import PopulationGenerator
from benchmarks.general_affine import GeneralAffine
from benchmarks.zonotope import Zonotope
from benchmarks.homothet import HomothetProjection
import timeit
import csv

approximation_type = {
    'G-Polymatroid': Aggregator,
    'General Affine': GeneralAffine,
    'Zonotope': Zonotope,
    # 'Homothet': HomothetProjection
}

def time_lp(approximation, N, T):
    population = PopulationGenerator(T, e2s_count=N)
    c = np.random.uniform(-1,1, size=T)
    approximation(population).solve_lp(c)

# N = 50
# Ts = np.arange(0, 96, 12) + 12
# n_runs = 10

N = 4
Ts = np.array([4,6,8,10])
n_runs = 2


with open(f'papers/TOSG/numerical_results/data/compVt.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['benchmark', 'T', 'time'])
    for T in Ts:
        for name, approximation in approximation_type.items():
            print(name)
            time = timeit.timeit(lambda: time_lp(approximation, N, T), number=n_runs)
            avg_time = time / n_runs
            writer.writerow([name, T, avg_time])
            print(f'{name} {T} {time}')