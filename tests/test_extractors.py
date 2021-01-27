#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import numpy as np

from miplearn.extractors import InstanceFeaturesExtractor
from miplearn.problems.knapsack import KnapsackInstance
from miplearn.solvers.learning import LearningSolver


def _get_instances():
    instances = [
        KnapsackInstance(
            weights=[1.0, 2.0, 3.0],
            prices=[10.0, 20.0, 30.0],
            capacity=2.5,
        ),
        KnapsackInstance(
            weights=[3.0, 4.0, 5.0],
            prices=[20.0, 30.0, 40.0],
            capacity=4.5,
        ),
    ]
    models = [instance.to_model() for instance in instances]
    solver = LearningSolver()
    for (i, instance) in enumerate(instances):
        solver.solve(instances[i], models[i])
    return instances, models


def test_instance_features_extractor():
    instances, models = _get_instances()
    features = InstanceFeaturesExtractor().extract(instances)
    assert features.shape == (2, 3)
