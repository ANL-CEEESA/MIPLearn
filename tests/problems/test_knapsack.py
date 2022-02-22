#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from scipy.stats import uniform, randint

from miplearn import LearningSolver
from miplearn.problems.knapsack import MultiKnapsackGenerator, MultiKnapsackInstance


def test_knapsack_generator() -> None:
    gen = MultiKnapsackGenerator(
        n=randint(low=100, high=101),
        m=randint(low=30, high=31),
        w=randint(low=0, high=1000),
        K=randint(low=500, high=501),
        u=uniform(loc=1.0, scale=1.0),
        alpha=uniform(loc=0.50, scale=0.0),
    )
    data = gen.generate(100)
    w_sum = sum(d.weights for d in data) / len(data)
    b_sum = sum(d.capacities for d in data) / len(data)
    assert round(float(np.mean(w_sum)), -1) == 500.0
    assert round(float(np.mean(b_sum)), -3) == 25000.0


def test_knapsack() -> None:
    data = MultiKnapsackGenerator(
        n=randint(low=5, high=6),
        m=randint(low=5, high=6),
    ).generate(1)
    instance = MultiKnapsackInstance(
        prices=data[0].prices,
        capacities=data[0].capacities,
        weights=data[0].weights,
    )
    solver = LearningSolver()
    solver.solve(instance)
