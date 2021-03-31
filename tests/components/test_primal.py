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
from miplearn.types import TrainingSample


def test_xy_sample_with_lp_solution() -> None:
    instance = cast(Instance, Mock(spec=Instance))
    instance.get_variable_category = Mock(  # type: ignore
        side_effect=lambda var_name, index: {
            0: "default",
            1: None,
            2: "default",
            3: "default",
        }[index]
    )
    instance.get_variable_features = Mock(  # type: ignore
        side_effect=lambda var, index: {
            0: [0.0, 0.0],
            1: [0.0, 1.0],
            2: [1.0, 0.0],
            3: [1.0, 1.0],
        }[index]
    )
    sample: TrainingSample = {
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
                0: 0.1,
                1: 0.1,
                2: 0.1,
                3: 0.1,
            }
        },
    }
    x_expected = {
        "default": np.array(
            [
                [0.0, 0.0, 0.1],
                [1.0, 0.0, 0.1],
                [1.0, 1.0, 0.1],
            ]
        )
    }
    y_expected = {
        "default": np.array(
            [
                [True, False],
                [False, True],
                [True, False],
            ]
        )
    }
    x_actual, y_actual = PrimalSolutionComponent.xy_sample(instance, sample)
    assert len(x_actual.keys()) == 1
    assert len(y_actual.keys()) == 1
    assert_array_equal(x_actual["default"], x_expected["default"])
    assert_array_equal(y_actual["default"], y_expected["default"])


def test_xy_sample_without_lp_solution() -> None:
    comp = PrimalSolutionComponent()
    instance = cast(Instance, Mock(spec=Instance))
    instance.get_variable_category = Mock(  # type: ignore
        side_effect=lambda var_name, index: {
            0: "default",
            1: None,
            2: "default",
            3: "default",
        }[index]
    )
    instance.get_variable_features = Mock(  # type: ignore
        side_effect=lambda var, index: {
            0: [0.0, 0.0],
            1: [0.0, 1.0],
            2: [1.0, 0.0],
            3: [1.0, 1.0],
        }[index]
    )
    sample: TrainingSample = {
        "Solution": {
            "x": {
                0: 0.0,
                1: 1.0,
                2: 1.0,
                3: 0.0,
            }
        },
    }
    x_expected = {
        "default": np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
    }
    y_expected = {
        "default": np.array(
            [
                [True, False],
                [False, True],
                [True, False],
            ]
        )
    }
    x_actual, y_actual = comp.xy_sample(instance, sample)
    assert len(x_actual.keys()) == 1
    assert len(y_actual.keys()) == 1
    assert_array_equal(x_actual["default"], x_expected["default"])
    assert_array_equal(y_actual["default"], y_expected["default"])


def test_predict() -> None:
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
    thr = Mock(spec=Threshold)
    thr.predict = Mock(return_value=[0.75, 0.75])
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
    instance.features = {
        "Variables": {
            "x": {
                0: None,
                1: None,
                2: None,
            }
        }
    }
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
    x = {
        "default": np.array(
            [
                [0.0, 0.0, 0.1],
                [0.0, 2.0, 0.5],
                [2.0, 0.0, 0.9],
            ]
        )
    }
    comp = PrimalSolutionComponent()
    comp.classifiers = {"default": clf}
    comp.thresholds = {"default": thr}
    solution_actual = comp.predict(instance)
    clf.predict_proba.assert_called_once()
    thr.predict.assert_called_once()
    assert_array_equal(x["default"], clf.predict_proba.call_args[0][0])
    assert_array_equal(x["default"], thr.predict.call_args[0][0])
    assert solution_actual == {
        "x": {
            0: 0.0,
            1: None,
            2: 1.0,
        }
    }
