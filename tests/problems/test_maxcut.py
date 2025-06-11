#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2025, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import random

import numpy as np

from miplearn.problems.maxcut import MaxCutGenerator, build_maxcut_model_gurobipy
from scipy.stats import randint, uniform

def _set_seed():
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
    assert list(data[0].graph.edges()) == [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4)]
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
    )
    data = gen.generate(3)
    assert len(data) == 3
    assert list(data[0].graph.nodes()) == [0, 1, 2, 3, 4]
    assert list(data[0].graph.edges()) == [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4)]
    assert data[0].weights.tolist() == [-1, 1, -1, -1, -1, 1]
    assert list(data[1].graph.nodes()) == [0, 1, 2, 3, 4]
    assert list(data[1].graph.edges()) == [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4)]
    assert data[1].weights.tolist() == [-1, -1, -1, 1, -1, -1]

def test_maxcut_model():
    _set_seed()
    data = MaxCutGenerator(
        n=randint(low=20, high=21),
        p=uniform(loc=0.5, scale=0.0),
        fix_graph=True,
    ).generate(1)[0]
    model = build_maxcut_model_gurobipy(data)
    model.optimize()
    assert model.inner.ObjVal == -26
