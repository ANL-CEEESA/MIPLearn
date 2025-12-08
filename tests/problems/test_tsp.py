#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from miplearn.problems.tsp import (
    TravelingSalesmanData,
    TravelingSalesmanGenerator,
    build_tsp_model_gurobipy,
)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import randint, uniform


def test_tsp_generator() -> None:
    np.random.seed(42)
    gen = TravelingSalesmanGenerator(
        x=uniform(loc=0.0, scale=1000.0),
        y=uniform(loc=0.0, scale=1000.0),
        n=randint(low=5, high=6),
        gamma=uniform(loc=1.0, scale=0.25),
        round=True,
    )
    data = gen.generate(1)
    assert data[0].distances.tolist() == [
        [0.0, 525.0, 950.0, 392.0, 382.0],
        [525.0, 0.0, 752.0, 761.0, 178.0],
        [950.0, 752.0, 0.0, 809.0, 721.0],
        [392.0, 761.0, 809.0, 0.0, 700.0],
        [382.0, 178.0, 721.0, 700.0, 0.0],
    ]

    model = build_tsp_model_gurobipy(data[0])
    model.optimize()
    assert model.inner.getAttr("x", model.inner.getVars()) == [
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]
