# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import LearningSolver
from miplearn.problems.stab import MaxStableSetInstance, MaxStableSetGenerator
import networkx as nx
import numpy as np


def test_stab():
    graph = nx.cycle_graph(5)
    weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    instance = MaxStableSetInstance(graph, weights)
    solver = LearningSolver()
    solver.solve(instance)
    assert instance.model.OBJ() == 8.0
    
    
def test_stab_generator():
    graph = nx.cycle_graph(5)
    base_weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    instances = MaxStableSetGenerator(graph=graph,
                                       base_weights=base_weights,
                                       perturbation_scale=1.0,
                                      ).generate(100_000)
    weights = np.array([instance.weights for instance in instances])
    weights_avg = np.round(np.average(weights, axis=0), 2)
    weights_std = np.round(np.std(weights, axis=0), 2)
    assert list(weights_avg) == [1.50, 2.50, 3.50, 4.50, 5.50]
    assert list(weights_std) == [0.29] * 5
