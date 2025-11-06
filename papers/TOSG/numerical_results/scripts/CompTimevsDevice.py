import numpy as np
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.population_generator import PopulationGenerator
import timeit
import csv

def time_lp(T, population):
    c = np.random.uniform(-1,1, size=T)
    pop = population(T)    
    agg = Aggregator(pop)
    agg.greedy(c)



population_type = {
    'ESS_single':   lambda T: PopulationGenerator(T, e1s_count=1),
    'V1G_single':   lambda T: PopulationGenerator(T, v1g_count=1),
    'DER_single':   lambda T: PopulationGenerator(T, der_count=1),
    'ESS':          lambda T: PopulationGenerator(T, e1s_count=200),
    'V1G':          lambda T: PopulationGenerator(T, v1g_count=200),
    'DER (200)':    lambda T: PopulationGenerator(T, der_count=200),
    'DER (1000)':   lambda T: PopulationGenerator(T, der_count=1000)
    }

# Ts = np.arange(0, 96, 12) + 12
# n_runs = 10

Ts = np.array([5,10,15,20])
n_runs = 10

with open(f'papers/TOSG/numerical_results/data/compVdevice.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['benchmark', 'T', 'time'])
    for name, population_generator in population_type.items():
        for T in Ts:
            time = timeit.timeit(lambda: time_lp(T, population_generator), number=n_runs)
            avg_time = time / n_runs
            writer.writerow([name, T, avg_time])
            print(f'{name} {T} {time}')
