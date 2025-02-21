import numpy as np
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.population_sampling import PopulationGenerator
from numerical_results.benchmarks.general_affine import GeneralAffine
from numerical_results.benchmarks.zonotope import Zonotope
from numerical_results.benchmarks.homothet import HomothetProjection
from flexitroid.utils.cost import generate_energy_price_curve
import csv

T = 24
N_households = 100

i = 0
while True:
    i+=1
    with open(f'data/case_study{T}_{N_households}.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        print(i, end='\r')
        run_id = np.random.randint(100000000)
        population = PopulationGenerator(T,
                                v2g_count=10,
                                v1g_count=10,
                                pv_count=40,
                                e2s_count=40,
                                )
        c = generate_energy_price_curve(T)
        print(f'Base line: {i}')
        base_profile = population.base_line_consumption()
        writer.writerow(['base_line', run_id, 'lp', c@base_profile])
        writer.writerow(['base_line', run_id, 'l_inf', np.max(base_profile)])

        print(f'g-polymatroid: {i}')

        g_polymatroid = Aggregator(population)
        g_polymatroid_lp = g_polymatroid.solve_linear_program(c)
        g_polymatroid_l_inf = g_polymatroid.solve_l_inf().solution

        writer.writerow(['g-polymatroid', run_id, 'lp', c@g_polymatroid_lp])
        writer.writerow(['g-polymatroid', run_id, 'l_inf', np.max(g_polymatroid_l_inf)])

        print(f'general affine: {i}')
        general_affine = GeneralAffine(population)
        general_affine.solve_linear_program(c)
        general_affine.solve_l_inf()
        writer.writerow(['general_affine', run_id, 'lp', c@general_affine.lp_x])
        writer.writerow(['general_affine', run_id, 'l_inf', general_affine.l_inf_t])

        # print(f'homothet: {i}')
        # homothet_projection = HomothetProjection(population)
        # homothet_projection.solve_linear_program(c)
        # homothet_projection.solve_l_inf()
        # writer.writerow(['homothet', run_id, 'lp', c@homothet_projection.lp_x])
        # writer.writerow(['homothet', run_id, 'l_inf', homothet_projection.l_inf_t])


        print(f'zonotope: {i}')
        zonotope = Zonotope(population)
        zonotope.solve_linear_program(c)
        zonotope.solve_l_inf()
        writer.writerow(['zonotope', run_id, 'lp', c@zonotope.lp_x])
        writer.writerow(['zonotope', run_id, 'l_inf', zonotope.l_inf_t])




