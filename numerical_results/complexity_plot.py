import numpy as np
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.population_sampling import PopulationGenerator
import timeit
import csv

def time_lp(T, population):
    c = np.random.uniform(-1,1, size=T)
    pop = population(T)    
    agg = Aggregator(pop)
    agg.solve_linear_program(c)

def generate_der_population(T):
    pop = PopulationGenerator(T, der_count=1000)
    return pop

def generate_v1g_population(T):
    pop = PopulationGenerator(T, v1g_count=1000)
    return pop

def generate_e1s_population(T):
    pop = PopulationGenerator(T, e1s_count=1000)
    return pop

def generate_pv_population(T):
    pop = PopulationGenerator(T, pv_count=1000)
    return pop

population_type = {
    'der': generate_der_population,
    'v1g': generate_v1g_population,
    'e1s': generate_e1s_population,
    'pv': generate_pv_population
}

Ts = np.arange(0, 96, 12) + 12

n_runs = 10
with open(f'data/population_time.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['population_type', 'T', 'time'])
    for name, population_generator in population_type.items():
        for T in Ts:
            time = timeit.timeit(lambda: time_lp(T, population_generator), number=n_runs)
            avg_time = time / n_runs
            writer.writerow([name, T, avg_time])
            print(f'{name} {T} {time}')
