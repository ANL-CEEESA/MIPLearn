# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import LearningSolver, BenchmarkRunner
from miplearn.warmstart import KnnWarmStartPredictor
from miplearn.problems.stab import MaxStableSetInstance, MaxStableSetGenerator
import networkx as nx
import numpy as np
import pyomo.environ as pe


def test_benchmark():
    graph = nx.cycle_graph(10)
    base_weights = np.random.rand(10)
    
    # Generate training and test instances
    train_instances = MaxStableSetGenerator(graph=graph,
                                            base_weights=base_weights,
                                            perturbation_scale=1.0,
                                           ).generate(5)
    
    test_instances  = MaxStableSetGenerator(graph=graph,
                                            base_weights=base_weights,
                                            perturbation_scale=1.0,
                                           ).generate(3)

    # Training phase...
    training_solver = LearningSolver()
    training_solver.parallel_solve(train_instances, n_jobs=10)
    training_solver.save("data.bin")

    # Test phase...
    test_solvers = {
        "Strategy A": LearningSolver(ws_predictor=None),
        "Strategy B": LearningSolver(ws_predictor=None),
    }
    benchmark = BenchmarkRunner(test_solvers)
    benchmark.load_fit("data.bin")
    benchmark.parallel_solve(test_instances, n_jobs=2)
    print(benchmark.raw_results())
