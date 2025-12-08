#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from scipy.stats import uniform, randint

from miplearn.problems.pmedian import PMedianGenerator, build_pmedian_model_gurobipy


def test_pmedian() -> None:
    np.random.seed(42)
    gen = PMedianGenerator(
        x=uniform(loc=0.0, scale=100.0),
        y=uniform(loc=0.0, scale=100.0),
        n=randint(low=5, high=6),
        p=randint(low=2, high=3),
        demands=uniform(loc=0, scale=20),
        capacities=uniform(loc=0, scale=100),
    )
    data = gen.generate(1)

    assert data[0].p == 2
    assert data[0].demands.tolist() == [0.41, 19.4, 16.65, 4.25, 3.64]
    assert data[0].capacities.tolist() == [18.34, 30.42, 52.48, 43.19, 29.12]
    assert data[0].distances.tolist() == [
        [0.0, 50.17, 82.42, 32.76, 33.2],
        [50.17, 0.0, 72.64, 72.51, 17.06],
        [82.42, 72.64, 0.0, 71.69, 70.92],
        [32.76, 72.51, 71.69, 0.0, 56.56],
        [33.2, 17.06, 70.92, 56.56, 0.0],
    ]

    model = build_pmedian_model_gurobipy(data[0])
    assert model.inner.numVars == 30
    assert model.inner.numConstrs == 11
    model.optimize()
    assert round(model.inner.objVal) == 107


def test_pmedian_generator_callable() -> None:
    np.random.seed(42)
    gen = PMedianGenerator(
        x=uniform(loc=0.0, scale=100.0),
        y=uniform(loc=0.0, scale=100.0),
        n=randint(low=10, high=11),
        p=lambda n: n // 5,
        demands=uniform(loc=0, scale=20),
        capacities=uniform(loc=0, scale=100),
    )
    data = gen.generate(1)
    assert data[0].p == 2
    assert len(data[0].demands) == 10
