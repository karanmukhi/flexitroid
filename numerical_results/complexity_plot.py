import numpy as np
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.population_generator import PopulationGenerator
import timeit
import csv

def time_lp(T, population):
    c = np.random.uniform(-1,1, size=T)
    pop = population(T)    
    agg = Aggregator(pop)
    agg.solve_linear_program(c)

def generate_der_population200(T):
    pop = PopulationGenerator(T, der_count=200)
    return pop

def generate_der_population1000(T):
    pop = PopulationGenerator(T, der_count=1000)
    return pop

def generate_v1g_population(T):
    pop = PopulationGenerator(T, v1g_count=1000)
    return pop

def generate_e1s_population(T):
    pop = PopulationGenerator(T, e1s_count=1000)
    return pop


population_type = {
    'ESS': generate_e1s_population,
    'V1G': generate_v1g_population,
    'DER (200)': generate_der_population200,
    'DER (1000)': generate_der_population1000,
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
