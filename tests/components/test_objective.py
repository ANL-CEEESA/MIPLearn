#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import cast
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from miplearn.instance import Instance
from miplearn.classifiers import Regressor
from miplearn.components.objective import ObjectiveValueComponent
from miplearn.types import TrainingSample
from tests.fixtures.knapsack import get_test_pyomo_instances


def test_xy_sample() -> None:
    instance = cast(Instance, Mock(spec=Instance))
    instance.features = {
        "Instance": {
            "User features": [1.0, 2.0],
        }
    }
    sample: TrainingSample = {
        "Lower bound": 1.0,
        "Upper bound": 2.0,
        "LP value": 3.0,
    }
    x_expected = {
        "Lower bound": [[1.0, 2.0, 3.0]],
        "Upper bound": [[1.0, 2.0, 3.0]],
    }
    y_expected = {
        "Lower bound": [[1.0]],
        "Upper bound": [[2.0]],
    }
    x_actual, y_actual = ObjectiveValueComponent.xy_sample(instance, sample)
    assert x_actual == x_expected
    assert y_actual == y_expected


def test_x_y_predict() -> None:
    # Construct instance
    instance = cast(Instance, Mock(spec=Instance))
    instance.get_instance_features = Mock(  # type: ignore
        return_value=[1.0, 2.0],
    )
    instance.training_data = [
        {
            "Lower bound": 1.0,
            "Upper bound": 2.0,
            "LP value": 3.0,
        },
        {
            "Lower bound": 1.5,
            "Upper bound": 2.2,
            "LP value": 3.4,
        },
    ]

    # Construct mock regressors
    lb_regressor = Mock(spec=Regressor)
    lb_regressor.predict = Mock(return_value=np.array([[5.0], [6.0]]))
    ub_regressor = Mock(spec=Regressor)
    ub_regressor.predict = Mock(return_value=np.array([[3.0], [3.0]]))
    comp = ObjectiveValueComponent(
        lb_regressor=lambda: lb_regressor,
        ub_regressor=lambda: ub_regressor,
    )

    # Should build x correctly
    x_expected = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.4]])
    assert_array_equal(comp.x([instance]), x_expected)

    # Should build y correctly
    y_actual = comp.y([instance])
    y_expected_lb = np.array([[1.0], [1.5]])
    y_expected_ub = np.array([[2.0], [2.2]])
    assert_array_equal(y_actual["Lower bound"], y_expected_lb)
    assert_array_equal(y_actual["Upper bound"], y_expected_ub)

    # Should pass arrays to regressors
    comp.fit([instance])
    assert_array_equal(lb_regressor.fit.call_args[0][0], x_expected)
    assert_array_equal(lb_regressor.fit.call_args[0][1], y_expected_lb)
    assert_array_equal(ub_regressor.fit.call_args[0][0], x_expected)
    assert_array_equal(ub_regressor.fit.call_args[0][1], y_expected_ub)

    # Should return predictions
    pred = comp.predict([instance])
    assert_array_equal(lb_regressor.predict.call_args[0][0], x_expected)
    assert_array_equal(ub_regressor.predict.call_args[0][0], x_expected)
    assert pred == {
        "Lower bound": [5.0, 6.0],
        "Upper bound": [3.0, 3.0],
    }


def test_obj_evaluate():
    instances, models = get_test_pyomo_instances()
    reg = Mock(spec=Regressor)
    reg.predict = Mock(return_value=np.array([[1000.0], [1000.0]]))
    comp = ObjectiveValueComponent(
        lb_regressor=lambda: reg,
        ub_regressor=lambda: reg,
    )
    comp.fit(instances)
    ev = comp.evaluate(instances)
    assert ev == {
        "Lower bound": {
            "Explained variance": 0.0,
            "Max error": 183.0,
            "Mean absolute error": 126.5,
            "Mean squared error": 19194.5,
            "Median absolute error": 126.5,
            "R2": -5.012843605607331,
        },
        "Upper bound": {
            "Explained variance": 0.0,
            "Max error": 183.0,
            "Mean absolute error": 126.5,
            "Mean squared error": 19194.5,
            "Median absolute error": 126.5,
            "R2": -5.012843605607331,
        },
    }
