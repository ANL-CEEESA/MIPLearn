# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import LearningSolver, BenchmarkRunner
from miplearn.warmstart import KnnWarmStartPredictor
from miplearn.problems.stab import MaxWeightStableSetGenerator
from scipy.stats import randint
import numpy as np
import pyomo.environ as pe
import os.path


def test_benchmark():
    # Generate training and test instances
    train_instances = MaxWeightStableSetGenerator(n=randint(low=25, high=26)).generate(5)    
    test_instances  = MaxWeightStableSetGenerator(n=randint(low=25, high=26)).generate(3)

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
    benchmark.parallel_solve(test_instances, n_jobs=2, n_trials=2)
    assert benchmark.raw_results().values.shape == (12,12)
    
    benchmark.save_results("/tmp/benchmark.csv")
    assert os.path.isfile("/tmp/benchmark.csv")
    
    benchmark = BenchmarkRunner(test_solvers)
    benchmark.load_results("/tmp/benchmark.csv")
    assert benchmark.raw_results().values.shape == (12,12)
