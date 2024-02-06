#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from scipy.stats import uniform, randint

from miplearn.problems.multiknapsack import (
    MultiKnapsackGenerator,
    MultiKnapsackData,
    build_multiknapsack_model_gurobipy,
)


def test_knapsack_generator() -> None:
    np.random.seed(42)
    gen = MultiKnapsackGenerator(
        n=randint(low=5, high=6),
        m=randint(low=3, high=4),
        w=randint(low=0, high=1000),
        K=randint(low=500, high=501),
        u=uniform(loc=0.0, scale=1.0),
        alpha=uniform(loc=0.25, scale=0.0),
        fix_w=True,
        w_jitter=uniform(loc=0.9, scale=0.2),
        p_jitter=uniform(loc=0.9, scale=0.2),
        round=True,
    )
    data = gen.generate(2)
    assert data[0].prices.tolist() == [433.0, 477.0, 802.0, 494.0, 458.0]
    assert data[0].capacities.tolist() == [458.0, 357.0, 392.0]
    assert data[0].weights.tolist() == [
        [111.0, 392.0, 945.0, 276.0, 108.0],
        [64.0, 633.0, 20.0, 602.0, 110.0],
        [510.0, 203.0, 303.0, 469.0, 85.0],
    ]

    assert data[1].prices.tolist() == [344.0, 527.0, 658.0, 519.0, 460.0]
    assert data[1].capacities.tolist() == [449.0, 377.0, 380.0]
    assert data[1].weights.tolist() == [
        [92.0, 473.0, 871.0, 264.0, 96.0],
        [67.0, 664.0, 21.0, 628.0, 129.0],
        [436.0, 209.0, 309.0, 481.0, 86.0],
    ]


def test_knapsack_model() -> None:
    data = MultiKnapsackData(
        prices=np.array([344.0, 527.0, 658.0, 519.0, 460.0]),
        capacities=np.array([449.0, 377.0, 380.0]),
        weights=np.array(
            [
                [92.0, 473.0, 871.0, 264.0, 96.0],
                [67.0, 664.0, 21.0, 628.0, 129.0],
                [436.0, 209.0, 309.0, 481.0, 86.0],
            ]
        ),
    )
    model = build_multiknapsack_model_gurobipy(data)
    model.optimize()
    assert model.inner.objVal == -460.0
