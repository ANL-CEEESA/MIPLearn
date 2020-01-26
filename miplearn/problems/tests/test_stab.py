# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import LearningSolver
from miplearn.problems.stab import MaxWeightStableSetInstance
from miplearn.problems.stab import MaxWeightStableSetGenerator
import networkx as nx
import numpy as np
from scipy.stats import uniform, randint


def test_stab():
    graph = nx.cycle_graph(5)
    weights = [1., 2., 3., 4., 5.]
    instance = MaxWeightStableSetInstance(graph, weights)
    solver = LearningSolver()
    solver.solve(instance)
    assert instance.model.OBJ() == 8.
    
    
def test_stab_generator_fixed_graph():
    from miplearn.problems.stab import MaxWeightStableSetGenerator
    gen = MaxWeightStableSetGenerator(w=uniform(loc=50., scale=10.),
                                      n=randint(low=10, high=11),
                                      density=uniform(loc=0.05, scale=0.),
                                      fix_graph=True)
    instances = gen.generate(1_000)
    weights = np.array([instance.weights for instance in instances])
    weights_avg_actual = np.round(np.average(weights, axis=0))
    weights_avg_expected = [55.0] * 10
    assert list(weights_avg_actual) == weights_avg_expected
    
    
def test_stab_generator_random_graph():
    from miplearn.problems.stab import MaxWeightStableSetGenerator
    gen = MaxWeightStableSetGenerator(w=uniform(loc=50., scale=10.),
                                      n=randint(low=30, high=41),
                                      density=uniform(loc=0.5, scale=0.),
                                      fix_graph=False)
    instances = gen.generate(1_000)
    n_nodes = [instance.graph.number_of_nodes() for instance in instances]
    n_edges = [instance.graph.number_of_edges() for instance in instances]
    assert np.round(np.mean(n_nodes)) == 35.
    assert np.round(np.mean(n_edges), -1) == 300.
