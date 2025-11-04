import pytest
import numpy as np
import cvxpy as cp
from flexitroid.devices.pv import PV
from flexitroid.devices.level1 import E1S, V1G
from flexitroid.devices.general_der import GeneralDER
from flexitroid.devices.level2 import E2S, V2G
from flexitroid.utils.population_generator import PopulationGenerator
from flexitroid.aggregations.v1g_aggregator import V1GAggregator
from flexitroid.aggregations.e1s_aggregator import E1SAggregator
from flexitroid.aggregations.pv_aggregator import PVAggregator
from flexitroid.aggregations.aggregator import Aggregator


def test_aggregation1():
    T = 30
    population = PopulationGenerator(
        T,
        pv_count=10,
        der_count=10,
        v1g_count=10,
        e1s_count=10,
        e2s_count=10,
        v2g_count=10,
    )
    aggregation = Aggregator(population)
    aggregation_vertex_tester(aggregation, population.device_list)


def test_aggregation2():
    T = 5
    population = PopulationGenerator(
        T,
        pv_count=0,
        der_count=0,
        v1g_count=10,
        e1s_count=10,
        e2s_count=0,
        v2g_count=0,
    )
    aggregation = Aggregator(population)
    aggregation_vertex_tester(aggregation, population.device_list)


def test_aggregation3():
    T = 30
    population = PopulationGenerator(
        T,
        pv_count=10,
        der_count=0,
        v1g_count=0,
        e1s_count=10,
        e2s_count=0,
        v2g_count=0,
    )
    aggregation = Aggregator(population)
    aggregation_vertex_tester(aggregation, population.device_list)


def test_der_aggregation():
    T = 5
    population = PopulationGenerator(T, der_count=10)
    aggregation = Aggregator(population)
    aggregation_vertex_tester(aggregation, population.device_list)


def test_pv_aggregation():
    population = [PV.example() for _ in range(100)]
    aggregation = PVAggregator(population)
    aggregation_vertex_tester(aggregation, population)


def test_e1s_aggregation():
    population = [E1S.example() for _ in range(100)]
    aggregation = E1SAggregator(population)
    aggregation_vertex_tester(aggregation, population)


def test_ev_aggregation():
    population = [V1G.example() for _ in range(100)]
    aggregation = V1GAggregator(population)
    aggregation_vertex_tester(aggregation, population)


def aggregation_vertex_tester(aggregation, population):
    T = aggregation.T
    c = np.random.uniform(-1, 1, size=T)
    agg_opt = aggregation.greedy(c)
    individual_opt = np.array([device.greedy(c) for device in population])
    agg_indiv = np.sum(individual_opt, axis=0)
    assert np.linalg.norm(agg_opt - agg_indiv) < 1e-6
