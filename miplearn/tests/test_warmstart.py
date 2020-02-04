# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import WarmStartComponent, LearningSolver
from miplearn.problems.knapsack import KnapsackInstance
import numpy as np
import tempfile


def _get_instances():
    return [
        KnapsackInstance(
            weights=[23., 26., 20., 18.],
            prices=[505., 352., 458., 220.],
            capacity=67.,
        ),
    ] * 2


def test_warm_start_save_load():
    state_file = tempfile.NamedTemporaryFile(mode="r")
    solver = LearningSolver(components={"warm-start": WarmStartComponent()})
    solver.parallel_solve(_get_instances(), n_jobs=2)
    solver.fit()
    comp = solver.components["warm-start"]
    assert comp.x_train["default"].shape == (8, 6)
    assert comp.y_train["default"].shape == (8, 2)
    assert "default" in comp.predictors.keys()
    solver.save_state(state_file.name)
    
    solver = LearningSolver(components={"warm-start": WarmStartComponent()})
    solver.load_state(state_file.name)
    comp = solver.components["warm-start"]
    assert comp.x_train["default"].shape == (8, 6)
    assert comp.y_train["default"].shape == (8, 2)
    assert "default" in comp.predictors.keys()
