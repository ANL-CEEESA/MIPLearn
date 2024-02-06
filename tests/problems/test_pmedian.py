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
        distances_jitter=uniform(loc=0.95, scale=0.1),
        demands_jitter=uniform(loc=0.95, scale=0.1),
        capacities_jitter=uniform(loc=0.95, scale=0.1),
        fixed=True,
    )
    data = gen.generate(2)

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

    assert data[1].p == 2
    assert data[1].demands.tolist() == [0.42, 19.03, 16.68, 4.27, 3.53]
    assert data[1].capacities.tolist() == [19.2, 31.26, 54.79, 44.9, 29.41]
    assert data[1].distances.tolist() == [
        [0.0, 51.6, 83.31, 33.77, 31.95],
        [51.6, 0.0, 70.25, 71.09, 17.05],
        [83.31, 70.25, 0.0, 68.81, 67.62],
        [33.77, 71.09, 68.81, 0.0, 58.88],
        [31.95, 17.05, 67.62, 58.88, 0.0],
    ]

    model = build_pmedian_model_gurobipy(data[0])
    assert model.inner.numVars == 30
    assert model.inner.numConstrs == 11
    model.optimize()
    assert round(model.inner.objVal) == 107
