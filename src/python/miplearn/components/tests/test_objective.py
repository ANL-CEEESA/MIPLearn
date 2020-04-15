#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from unittest.mock import Mock

import numpy as np
from miplearn import ObjectiveValueComponent
from miplearn.classifiers import Regressor
from miplearn.tests import get_training_instances_and_models


def test_usage():
    instances, models = get_training_instances_and_models()
    comp = ObjectiveValueComponent()
    comp.fit(instances)
    assert instances[0].lower_bound == 1183.0
    assert instances[0].upper_bound == 1183.0
    assert comp.predict(instances).tolist() == [[1183.0, 1183.0],
                                                [1070.0, 1070.0]]


def test_obj_evaluate():
    instances, models = get_training_instances_and_models()
    reg = Mock(spec=Regressor)
    reg.predict = Mock(return_value=np.array([[1000.0], [1000.0]]))
    comp = ObjectiveValueComponent(regressor=reg)
    comp.fit(instances)
    ev = comp.evaluate(instances)
    assert ev == {
        'Lower bound': {
            'Explained variance': 0.0,
            'Max error': 183.0,
            'Mean absolute error': 126.5,
            'Mean squared error': 19194.5,
            'Median absolute error': 126.5,
            'R2': -5.012843605607331,
        },
        'Upper bound': {
            'Explained variance': 0.0,
            'Max error': 183.0,
            'Mean absolute error': 126.5,
            'Mean squared error': 19194.5,
            'Median absolute error': 126.5,
            'R2': -5.012843605607331,
        }
    }
