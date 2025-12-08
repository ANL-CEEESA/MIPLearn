#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2025, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import random
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import randint, uniform

from miplearn.h5 import H5File
from miplearn.problems.maxcut import (
    MaxCutGenerator,
    build_maxcut_model_gurobipy,
    build_maxcut_model_pyomo,
)
from miplearn.solvers.abstract import AbstractModel


def _set_seed() -> None:
    random.seed(42)
    np.random.seed(42)


def test_maxcut_generator_not_fixed() -> None:
    _set_seed()
    gen = MaxCutGenerator(
        n=randint(low=5, high=6),
        p=uniform(loc=0.5, scale=0.0),
        fix_graph=False,
    )
    data = gen.generate(3)
    assert len(data) == 3
    assert list(data[0].graph.nodes()) == [0, 1, 2, 3, 4]
    assert list(data[0].graph.edges()) == [
        (0, 2),
        (0, 3),
        (0, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    assert data[0].weights.tolist() == [-1, 1, -1, -1, -1, 1]
    assert list(data[1].graph.nodes()) == [0, 1, 2, 3, 4]
    assert list(data[1].graph.edges()) == [(0, 1), (0, 3), (0, 4), (1, 4), (3, 4)]
    assert data[1].weights.tolist() == [-1, -1, -1, 1, -1]


def test_maxcut_generator_fixed() -> None:
    random.seed(42)
    np.random.seed(42)
    gen = MaxCutGenerator(
        n=randint(low=5, high=6),
        p=uniform(loc=0.5, scale=0.0),
        fix_graph=True,
        w_jitter=0.25,
    )
    data = gen.generate(3)
    assert len(data) == 3
    for i in range(3):
        assert list(data[i].graph.nodes()) == [0, 1, 2, 3, 4]
        assert list(data[i].graph.edges()) == [
            (0, 2),
            (0, 3),
            (0, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ]
    assert data[0].weights.tolist() == [-1, -1, 1, 1, -1, 1]
    assert data[1].weights.tolist() == [-1, -1, -1, -1, 1, -1]
    assert data[2].weights.tolist() == [1, 1, -1, -1, -1, 1]


def test_maxcut_model() -> None:
    _set_seed()
    data = MaxCutGenerator(
        n=randint(low=10, high=11),
        p=uniform(loc=0.5, scale=0.0),
        fix_graph=True,
    ).generate(1)[0]
    for model in [
        build_maxcut_model_gurobipy(data),
        build_maxcut_model_pyomo(data),
    ]:
        assert isinstance(model, AbstractModel)
        with TemporaryDirectory() as tempdir:
            with H5File(f"{tempdir}/data.h5", "w") as h5:
                model.extract_after_load(h5)
                obj_lin = h5.get_array("static_var_obj_coeffs")
                assert obj_lin is not None
                assert obj_lin.tolist() == [
                    3.0,
                    1.0,
                    3.0,
                    1.0,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                ]
                obj_quad = h5.get_array("static_var_obj_coeffs_quad")
                assert obj_quad is not None
                assert obj_quad.tolist() == [
                    [0.0, 0.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0],
                    [0.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
                model.optimize()
                model.extract_after_mip(h5)
                assert h5.get_scalar("mip_obj_value") == -4
