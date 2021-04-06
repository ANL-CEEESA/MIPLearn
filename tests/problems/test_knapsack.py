#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from scipy.stats import uniform, randint

from miplearn.problems.knapsack import MultiKnapsackGenerator


def test_knapsack_generator():
    gen = MultiKnapsackGenerator(
        n=randint(low=100, high=101),
        m=randint(low=30, high=31),
        w=randint(low=0, high=1000),
        K=randint(low=500, high=501),
        u=uniform(loc=1.0, scale=1.0),
        alpha=uniform(loc=0.50, scale=0.0),
    )
    instances = gen.generate(100)
    w_sum = sum(instance.weights for instance in instances) / len(instances)
    b_sum = sum(instance.capacities for instance in instances) / len(instances)
    assert round(np.mean(w_sum), -1) == 500.0
    assert round(np.mean(b_sum), -3) == 25000.0
