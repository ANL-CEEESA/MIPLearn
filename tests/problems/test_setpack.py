#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from scipy.stats import randint, uniform

from miplearn.problems.setpack import (
    SetPackData,
    SetPackGenerator,
    build_setpack_model_gurobipy,
)


def test_setpack() -> None:
    data = SetPackData(
        costs=np.array([5, 10, 12, 6, 8]),
        incidence_matrix=np.array(
            [
                [1, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1],
            ],
        ),
    )
    model = build_setpack_model_gurobipy(data)
    model.optimize()
    assert model.inner.objval == -22.0


def test_set_pack_generator_callable() -> None:
    np.random.seed(42)
    gen = SetPackGenerator(
        n_elements=randint(low=4, high=5),
        n_sets=lambda n: n * 2,
        costs=uniform(loc=10.0, scale=0.0),
        density=uniform(loc=0.5, scale=0),
        K=uniform(loc=0, scale=0),
    )
    data = gen.generate(1)
    n_elements, n_sets = data[0].incidence_matrix.shape
    assert n_elements == 4
    assert n_sets == 8
    assert len(data[0].costs) == 8
