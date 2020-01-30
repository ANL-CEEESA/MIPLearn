# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import LearningSolver
from miplearn.problems.knapsack import MultiKnapsackGenerator, MultiKnapsackInstance
from scipy.stats import uniform, randint
import numpy as np


def test_knapsack_generator():
    gen = MultiKnapsackGenerator(n=randint(low=100, high=101),
                                 m=randint(low=30, high=31),
                                 w=randint(low=0, high=1000),
                                 K=randint(low=500, high=501),
                                 u=uniform(loc=1.0, scale=1.0),
                                 alpha=uniform(loc=0.50, scale=0.0),
                                )
    instances = gen.generate(100)
    w_sum = sum(instance.weights for instance in instances) / len(instances)
    p_sum = sum(instance.prices for instance in instances) / len(instances)
    b_sum = sum(instance.capacities for instance in instances) / len(instances)
    assert round(np.mean(w_sum), -1) == 500.
    assert round(np.mean(p_sum), -1) == 1250.
    assert round(np.mean(b_sum), -3) == 25000.

    
def test_knapsack_instance():
    instance = MultiKnapsackInstance(
        prices=np.array([5.0, 10.0, 15.0]),
        capacities=np.array([20.0, 30.0]),
        weights=np.array([
            [5.0,  5.0,  5.0],
            [5.0, 10.0, 15.0],
        ])
    )
    
    assert (instance.get_instance_features() == np.array([
        5.0, 10.0, 15.0, 20.0, 30.0, 5.0, 5.0, 5.0, 5.0, 10.0, 15.0
    ])).all()
    
    solver = LearningSolver()
    results = solver.solve(instance)
    assert results["Problem"][0]["Lower bound"] == 30.0