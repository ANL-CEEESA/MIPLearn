#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from tempfile import TemporaryDirectory

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
        density=uniform(loc=0.5, scale=0),
        K=uniform(loc=25, scale=0),
    )
    data = gen.generate(1)
    assert data[0].costs.round(1).tolist() == [136.8, 86.2, 25.7, 27.3, 102.5]
    assert data[0].incidence_matrix.tolist() == [
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
        with TemporaryDirectory() as tempdir:
            with H5File(f"{tempdir}/data.h5", "w") as h5:
                model.optimize()
                model.extract_after_mip(h5)
                assert h5.get_scalar("mip_obj_value") == 11.0
