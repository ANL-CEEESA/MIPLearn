#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from scipy.stats import uniform, randint

from miplearn.problems.binpack import build_binpack_model, BinPackData, BinPackGenerator


def test_binpack_generator() -> None:
    np.random.seed(42)
    gen = BinPackGenerator(
        n=randint(low=10, high=11),
        sizes=uniform(loc=0, scale=10),
        capacity=uniform(loc=100, scale=0),
        sizes_jitter=uniform(loc=0.9, scale=0.2),
        capacity_jitter=uniform(loc=0.9, scale=0.2),
        fix_items=True,
    )
    data = gen.generate(2)
    assert data[0].sizes.tolist() == [
        3.39,
        10.4,
        7.81,
        5.64,
        1.46,
        1.46,
        0.56,
        8.7,
        5.93,
        6.79,
    ]
    assert data[0].capacity == 102.24
    assert data[1].sizes.tolist() == [
        3.48,
        9.11,
        7.12,
        5.93,
        1.65,
        1.47,
        0.58,
        8.82,
        5.47,
        7.23,
    ]
    assert data[1].capacity == 93.41


def test_binpack() -> None:
    model = build_binpack_model(
        BinPackData(
            sizes=np.array([4, 8, 1, 4, 2, 1]),
            capacity=10,
        )
    )
    model.optimize()
    assert model.inner.objVal == 2.0
