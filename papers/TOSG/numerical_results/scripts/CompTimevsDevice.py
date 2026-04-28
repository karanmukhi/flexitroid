import numpy as np
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.population_generator import PopulationGenerator
import timeit
import csv
import time
from matplotlib.ticker import MultipleLocator

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from papers.TOSG.color_config import OKABE_ITO_CYCLE, OKABE_ITO_PALETTE, OKABE_ITO_BENCHMARK_COLORS, OKABE_ITO_DEVICE_COLORS
from shutil import which
import csv
import pandas as pd
import numpy as np

def time_lp(T, population):
    c = np.random.uniform(-1,1, size=T)
    pop = population(T)    
    agg = Aggregator(pop)
    agg.greedy(c)


def time_lp(agg):
    c = np.random.uniform(-1,1, size=T)
    start = time.time()
    agg.greedy(c)
    end = time.time()
    return end - start

population_type = {
    'DER':         lambda T: PopulationGenerator(T, der_count=100),
    'EV':          lambda T: PopulationGenerator(T, v1g_count=10000),
    'DL':          lambda T: PopulationGenerator(T, e1s_count=10000),
    }


Ts = np.arange(0, 60, 6) + 6
n_runs = 1000

with open(f'papers/TOSG/numerical_results/data/compVdevice.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['benchmark', 'T', 'time'])
    for name, population_generator in population_type.items():
        if name == 'DER':
            n_runs = 10
        elif name == 'EV':
            n_runs = 100
        else: n_runs = 1000
        for T in Ts:
            pop = population_generator(T)    
            agg = Aggregator(pop)
            times = [time_lp(agg) for _ in range(n_runs)]
            # time = timeit.timeit(lambda: time_lp(T, population_generator), number=n_runs)
            # avg_time = time / n_runs
            if name == 'DER':
                times = np.array(times) * 100
                
            avg_time = np.mean(times)
            writer.writerow([name, T, avg_time])
            print(f'{name} {T} {avg_time}')
