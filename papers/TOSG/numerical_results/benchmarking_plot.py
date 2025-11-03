import numpy as np
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.population_generator import PopulationGenerator
from numerical_results.benchmarks.general_affine import GeneralAffine
from numerical_results.benchmarks.zonotope import Zonotope
from numerical_results.benchmarks.homothet import HomothetProjection
import timeit
import csv

approximation_type = {
    'G-Polymatroid': Aggregator,
    'General Affine': GeneralAffine,
    'Zonotope': Zonotope,
    'Homothet': HomothetProjection
}
def time_lp(approximation):
    T = 24
    population = PopulationGenerator(T, e2s_count=N)
    c = np.random.uniform(-1,1, size=T)
    approximation(population).solve_linear_program(c)

Ns = np.arange(0, 500, 50) + 50
n_runs = 10
with open(f'data/benchmarking_time.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['benchmark', 'N', 'time'])
    for N in Ns:
        for name, approximation in approximation_type.items():
            time = timeit.timeit(lambda: time_lp(approximation), number=n_runs)
            avg_time = time / n_runs
            writer.writerow([name, N, avg_time])
            print(f'{name} {N} {time}')