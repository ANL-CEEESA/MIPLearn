#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import cast, List
from unittest.mock import Mock, call

import numpy as np
from numpy.testing import assert_array_equal

from miplearn import Classifier
from miplearn.classifiers.threshold import Threshold, MinPrecisionThreshold
from miplearn.components.primal import PrimalSolutionComponent
from miplearn.instance import Instance
from tests import get_test_pyomo_instances


def test_x_y_fit() -> None:
    comp = PrimalSolutionComponent()
    training_instances = cast(
        List[Instance],
        [
            Mock(spec=Instance),
            Mock(spec=Instance),
        ],
    )

    # Construct first instance
    training_instances[0].get_variable_category = Mock(  # type: ignore
        side_effect=lambda var_name, index: {
            0: "default",
            1: None,
            2: "default",
            3: "default",
        }[index]
    )
    training_instances[0].get_variable_features = Mock(  # type: ignore
        side_effect=lambda var, index: {
            0: [0.0, 0.0],
            1: [0.0, 1.0],
            2: [1.0, 0.0],
            3: [1.0, 1.0],
        }[index]
    )
    training_instances[0].training_data = [
        {
            "Solution": {
                "x": {
                    0: 0.0,
                    1: 1.0,
                    2: 0.0,
                    3: 0.0,
                }
            },
            "LP solution": {
                "x": {
                    0: 0.1,
                    1: 0.1,
                    2: 0.1,
                    3: 0.1,
                }
            },
        },
        {
            "Solution": {
                "x": {
                    0: 0.0,
                    1: 1.0,
                    2: 1.0,
                    3: 0.0,
                }
            },
            "LP solution": {
                "x": {
                    0: 0.2,
                    1: 0.2,
                    2: 0.2,
                    3: 0.2,
                }
            },
        },
    ]

    # Construct second instance
    training_instances[1].get_variable_category = Mock(  # type: ignore
        side_effect=lambda var_name, index: {
            0: "default",
            1: None,
            2: "default",
            3: "default",
        }[index]
    )
    training_instances[1].get_variable_features = Mock(  # type: ignore
        side_effect=lambda var, index: {
            0: [0.0, 0.0],
            1: [0.0, 2.0],
            2: [2.0, 0.0],
            3: [2.0, 2.0],
        }[index]
    )
    training_instances[1].training_data = [
        {
            "Solution": {
                "x": {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                    3: 1.0,
                }
            },
            "LP solution": {
                "x": {
                    0: 0.3,
                    1: 0.3,
                    2: 0.3,
                    3: 0.3,
                }
            },
        },
        {
            "Solution": None,
            "LP solution": None,
        },
    ]

    # Test x
    x_expected = {
        "default": np.array(
            [
                [0.0, 0.0, 0.1],
                [1.0, 0.0, 0.1],
                [1.0, 1.0, 0.1],
                [0.0, 0.0, 0.2],
                [1.0, 0.0, 0.2],
                [1.0, 1.0, 0.2],
                [0.0, 0.0, 0.3],
                [2.0, 0.0, 0.3],
                [2.0, 2.0, 0.3],
            ]
        )
    }
    x_actual = comp.x(training_instances)
    assert len(x_actual.keys()) == 1
    assert_array_equal(x_actual["default"], x_expected["default"])

    # Test y
    y_expected = {
        "default": np.array(
            [
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [False, True],
                [True, False],
                [False, True],
                [False, True],
                [False, True],
            ]
        )
    }
    y_actual = comp.y(training_instances)
    assert len(y_actual.keys()) == 1
    assert_array_equal(y_actual["default"], y_expected["default"])

    # Test fit
    classifier = Mock(spec=Classifier)
    threshold = Mock(spec=Threshold)
    classifier_factory = Mock(return_value=classifier)
    threshold_factory = Mock(return_value=threshold)
    comp = PrimalSolutionComponent(
        classifier=classifier_factory,
        threshold=threshold_factory,
    )
    comp.fit(training_instances)

    # Should build and train classifier for "default" category
    classifier_factory.assert_called_once()
    assert_array_equal(x_actual["default"], classifier.fit.call_args.args[0])
    assert_array_equal(y_actual["default"], classifier.fit.call_args.args[1])

    # Should build and train threshold for "default" category
    threshold_factory.assert_called_once()
    assert classifier == threshold.fit.call_args.args[0]
    assert_array_equal(x_actual["default"], threshold.fit.call_args.args[1])
    assert_array_equal(y_actual["default"], threshold.fit.call_args.args[2])


def test_predict() -> None:
    comp = PrimalSolutionComponent()

    clf = Mock(spec=Classifier)
    clf.predict_proba = Mock(
        return_value=np.array(
            [
                [0.9, 0.1],
                [0.5, 0.5],
                [0.1, 0.9],
            ]
        )
    )
    comp.classifiers = {"default": clf}

    thr = Mock(spec=Threshold)
    thr.predict = Mock(return_value=[0.75, 0.75])
    comp.thresholds = {"default": thr}

    instance = cast(Instance, Mock(spec=Instance))
    instance.get_variable_category = Mock(  # type: ignore
        return_value="default",
    )
    instance.get_variable_features = Mock(  # type: ignore
        side_effect=lambda var, index: {
            0: [0.0, 0.0],
            1: [0.0, 2.0],
            2: [2.0, 0.0],
        }[index]
    )
    instance.training_data = [
        {
            "LP solution": {
                "x": {
                    0: 0.1,
                    1: 0.5,
                    2: 0.9,
                }
            }
        }
    ]

    x = comp.x([instance])
    solution_actual = comp.predict(instance)

    # Should ask for probabilities and thresholds
    clf.predict_proba.assert_called_once()
    thr.predict.assert_called_once()
    assert_array_equal(x["default"], clf.predict_proba.call_args.args[0])
    assert_array_equal(x["default"], thr.predict.call_args.args[0])

    assert solution_actual == {
        "x": {
            0: 0.0,
            1: None,
            2: 1.0,
        }
    }
