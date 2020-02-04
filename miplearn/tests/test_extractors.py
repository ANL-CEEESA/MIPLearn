# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn.problems.knapsack import KnapsackInstance
from miplearn import (UserFeaturesExtractor,
                      SolutionExtractor)
import numpy as np
import pyomo.environ as pe


def _get_instances():
    return [
        KnapsackInstance(weights=[1., 2., 3.],
                         prices=[10., 20., 30.],
                         capacity=2.5,
                        ),
        KnapsackInstance(weights=[3., 4., 5.],
                         prices=[20., 30., 40.],
                         capacity=4.5,
                        ),
    ]


def test_user_features():
    instances = _get_instances()
    extractor = UserFeaturesExtractor()
    features = extractor.extract(instances)
    assert isinstance(features, dict)
    assert "default" in features.keys()
    assert isinstance(features["default"], np.ndarray)
    assert features["default"].shape == (6, 4)

    
def test_solution_extractor():
    instances = _get_instances()
    models = [instance.to_model() for instance in instances]
    for model in models:
        solver = pe.SolverFactory("cbc")
        solver.solve(model)
    extractor = SolutionExtractor()
    features = extractor.extract(instances, models)
    assert isinstance(features, dict)
    assert "default" in features.keys()
    assert isinstance(features["default"], np.ndarray)
    assert features["default"].shape == (6, 2)
    assert features["default"].ravel().tolist() == [
        1., 0.,
        0., 1.,
        1., 0.,
        1., 0.,
        0., 1.,
        1., 0.,
    ]
