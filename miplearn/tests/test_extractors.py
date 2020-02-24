#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn.problems.knapsack import KnapsackInstance
from miplearn import (LearningSolver,
                      SolutionExtractor,
                      CombinedExtractor,
                      InstanceFeaturesExtractor,
                      VariableFeaturesExtractor,
                     )
import numpy as np
import pyomo.environ as pe


def _get_instances():
    instances = [
        KnapsackInstance(weights=[1., 2., 3.],
                         prices=[10., 20., 30.],
                         capacity=2.5,
                        ),
        KnapsackInstance(weights=[3., 4., 5.],
                         prices=[20., 30., 40.],
                         capacity=4.5,
                        ),
    ]
    models = [instance.to_model() for instance in instances]
    solver = LearningSolver()
    for (i, instance) in enumerate(instances):
        solver.solve(instances[i], models[i])
    return instances, models


def test_solution_extractor():
    instances, models = _get_instances()
    features = SolutionExtractor().extract(instances, models)
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

    
def test_combined_extractor():
    instances, models = _get_instances()
    extractor = CombinedExtractor(extractors=[VariableFeaturesExtractor(),
                                              SolutionExtractor()])
    features = extractor.extract(instances, models)
    assert isinstance(features, dict)
    assert "default" in features.keys()
    assert isinstance(features["default"], np.ndarray)
    assert features["default"].shape == (6, 7)
    
    
def test_instance_features_extractor():
    instances, models = _get_instances()
    features = InstanceFeaturesExtractor().extract(instances)
    assert features.shape == (2,3)
    
    
def test_variable_features_extractor():
    instances, models = _get_instances()
    features = VariableFeaturesExtractor().extract(instances)
    assert isinstance(features, dict)
    assert "default" in features
    assert features["default"].shape == (6,5)
    