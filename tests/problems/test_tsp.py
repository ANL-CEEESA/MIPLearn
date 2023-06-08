#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from miplearn.problems.tsp import (
    TravelingSalesmanData,
    TravelingSalesmanGenerator,
    build_tsp_model,
)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import randint, uniform


def test_tsp_generator() -> None:
    np.random.seed(42)
    gen = TravelingSalesmanGenerator(
        x=uniform(loc=0.0, scale=1000.0),
        y=uniform(loc=0.0, scale=1000.0),
        n=randint(low=3, high=4),
        gamma=uniform(loc=1.0, scale=0.25),
        fix_cities=True,
        round=True,
    )
    data = gen.generate(2)
    assert data[0].distances.tolist() == [
        [0.0, 591.0, 996.0],
        [591.0, 0.0, 765.0],
        [996.0, 765.0, 0.0],
    ]
    assert data[1].distances.tolist() == [
        [0.0, 556.0, 853.0],
        [556.0, 0.0, 779.0],
        [853.0, 779.0, 0.0],
    ]


def test_tsp() -> None:
    data = TravelingSalesmanData(
        n_cities=6,
        distances=squareform(
            pdist(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 1.0],
                    [3.0, 1.0],
                ]
            )
        ),
    )
    model = build_tsp_model(data)
    model.optimize()
    assert model.inner.getAttr("x", model.inner.getVars()) == [
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
    ]
