#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from scipy.stats import uniform, randint

from miplearn.problems.binpack import (
    build_binpack_model_gurobipy,
    BinPackData,
    BinPackGenerator,
)


def test_binpack_generator() -> None:
    np.random.seed(42)
    gen = BinPackGenerator(
        n=randint(low=10, high=11),
        sizes=uniform(loc=0, scale=10),
        capacity=uniform(loc=100, scale=0),
    )
    data = gen.generate(2)
    assert data[0].sizes.tolist() == [
        3.75,
        9.51,
        7.32,
        5.99,
        1.56,
        1.56,
        0.58,
        8.66,
        6.01,
        7.08,
    ]
    assert data[0].capacity == 100.0
    assert data[1].sizes.tolist() == [
        0.21,
        9.7,
        8.32,
        2.12,
        1.82,
        1.83,
        3.04,
        5.25,
        4.32,
        2.91,
    ]
    assert data[1].capacity == 100.0


def test_binpack() -> None:
    model = build_binpack_model_gurobipy(
        BinPackData(
            sizes=np.array([4, 8, 1, 4, 2, 1]),
            capacity=10,
        )
    )
    model.optimize()
    assert model.inner.objVal == 2.0
