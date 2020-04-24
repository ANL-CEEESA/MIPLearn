#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from unittest.mock import Mock

import numpy as np
from miplearn import BranchPriorityComponent, BranchPriorityExtractor
from miplearn.classifiers import Regressor
from miplearn.tests import get_training_instances_and_models


def test_branch_extract():
    instances, models = get_training_instances_and_models()
    instances[0].branch_priorities = {"x": {0: 100, 1: 200, 2: 300, 3: 400}}
    instances[1].branch_priorities = {"x": {0: 150, 1: 250, 2: 350, 3: 450}}
    priorities = BranchPriorityExtractor().extract(instances)
    assert priorities["default"].tolist() == [100, 200, 300, 400, 150, 250, 350, 450]


def test_branch_calculate():
    instances, models = get_training_instances_and_models()
    comp = BranchPriorityComponent()

    # If instances do not have branch_priority property, fit should compute them
    comp.fit(instances)
    assert instances[0].branch_priorities == {"x": {0: 5730, 1: 24878, 2: 0, 3: 0,}}

    # If instances already have branch_priority, fit should not modify them
    instances[0].branch_priorities = {"x": {0: 100, 1: 200, 2: 300, 3: 400}}
    comp.fit(instances)
    assert instances[0].branch_priorities == {"x": {0: 100, 1: 200, 2: 300, 3: 400}}


def test_branch_x_y_predict():
    instances, models = get_training_instances_and_models()
    instances[0].branch_priorities = {"x": {0: 100, 1: 200, 2: 300, 3: 400}}
    instances[1].branch_priorities = {"x": {0: 150, 1: 250, 2: 350, 3: 450}}
    comp = BranchPriorityComponent()
    comp.regressors["default"] = Mock(spec=Regressor)
    comp.regressors["default"].predict = Mock(return_value=np.array([150., 100., 0., 0.]))
    x, y = comp.x(instances), comp.y(instances)
    assert x["default"].shape == (8, 5)
    assert y["default"].shape == (8,)
    pred = comp.predict(instances[0])
    assert pred == {"x": {0: 150., 1: 100., 2: 0., 3: 0.}}
