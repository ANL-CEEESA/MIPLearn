#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn import LearningSolver, PrimalSolutionComponent
from miplearn.problems.knapsack import KnapsackInstance
import numpy as np
import tempfile


def _get_instances():
    instances = [
        KnapsackInstance(
            weights=[23., 26., 20., 18.],
            prices=[505., 352., 458., 220.],
            capacity=67.,
        ),
    ] * 5
    models = [inst.to_model() for inst in instances]
    solver = LearningSolver()
    for i in range(len(instances)):
        solver.solve(instances[i], models[i])
    return instances, models


def test_predict():
    instances, models = _get_instances()
    comp = PrimalSolutionComponent()
    comp.fit(instances)
    solution = comp.predict(instances[0])
    assert "x" in solution
    for idx in range(4):
        assert idx in solution["x"]
