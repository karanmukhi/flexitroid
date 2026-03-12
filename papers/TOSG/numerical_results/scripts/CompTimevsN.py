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


# T = 24
# Ns = np.arange(0, 500, 50) + 50
# n_runs = 10

T = 4
Ns = np.array([2,3,5,6])
n_runs = 2


with open(f'papers/TOSG/numerical_results/data/compVn.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['benchmark', 'N', 'time'])
    for N in Ns:
        for name, approximation in approximation_type.items():
            print(name)
            time = timeit.timeit(lambda: time_lp(approximation, N, T), number=n_runs)
            avg_time = time / n_runs
            writer.writerow([name, N, avg_time])
            print(f'{name} {N} {time}')