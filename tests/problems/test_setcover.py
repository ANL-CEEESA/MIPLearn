#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from tempfile import NamedTemporaryFile

import numpy as np
from scipy.stats import randint, uniform

from miplearn.h5 import H5File
from miplearn.problems.setcover import (
    SetCoverData,
    build_setcover_model_gurobipy,
    SetCoverGenerator,
    build_setcover_model_pyomo,
)
from miplearn.solvers.abstract import AbstractModel


def test_set_cover_generator() -> None:
    np.random.seed(42)
    gen = SetCoverGenerator(
        n_elements=randint(low=3, high=4),
        n_sets=randint(low=5, high=6),
        costs=uniform(loc=0.0, scale=100.0),
        costs_jitter=uniform(loc=0.95, scale=0.10),
        density=uniform(loc=0.5, scale=0),
        K=uniform(loc=25, scale=0),
        fix_sets=False,
    )
    data = gen.generate(2)

    assert data[0].costs.round(1).tolist() == [136.8, 86.2, 25.7, 27.3, 102.5]
    assert data[0].incidence_matrix.tolist() == [
        [1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1],
    ]
    assert data[1].costs.round(1).tolist() == [63.5, 76.6, 48.1, 74.1, 93.3]
    assert data[1].incidence_matrix.tolist() == [
        [1, 1, 0, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0],
    ]


def test_set_cover_generator_with_fixed_sets() -> None:
    np.random.seed(42)
    gen = SetCoverGenerator(
        n_elements=randint(low=3, high=4),
        n_sets=randint(low=5, high=6),
        costs=uniform(loc=0.0, scale=100.0),
        costs_jitter=uniform(loc=0.95, scale=0.10),
        density=uniform(loc=0.5, scale=0.00),
        fix_sets=True,
    )
    data = gen.generate(3)

    assert data[0].costs.tolist() == [136.75, 86.17, 25.71, 27.31, 102.48]
    assert data[1].costs.tolist() == [135.38, 82.26, 26.92, 26.58, 98.28]
    assert data[2].costs.tolist() == [138.37, 85.15, 26.95, 27.22, 106.17]

    print(data[0].incidence_matrix)

    for i in range(3):
        assert data[i].incidence_matrix.tolist() == [
            [1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1],
        ]


def test_set_cover() -> None:
    data = SetCoverData(
        costs=np.array([5, 10, 12, 6, 8]),
        incidence_matrix=np.array(
            [
                [1, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1],
            ],
        ),
    )
    for model in [
        build_setcover_model_pyomo(data),
        build_setcover_model_gurobipy(data),
    ]:
        assert isinstance(model, AbstractModel)
        with NamedTemporaryFile() as tempfile:
            with H5File(tempfile.name) as h5:
                model.optimize()
                model.extract_after_mip(h5)
                assert h5.get_scalar("mip_obj_value") == 11.0
