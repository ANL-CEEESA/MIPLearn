#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import cast
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from miplearn import Classifier
from miplearn.classifiers.threshold import Threshold
from miplearn.components.primal import PrimalSolutionComponent
from miplearn.instance import Instance
from miplearn.types import TrainingSample, Features


def test_xy_sample_with_lp_solution() -> None:
    features: Features = {
        "Variables": {
            "x": {
                0: {
                    "Category": "default",
                    "User features": [0.0, 0.0],
                },
                1: {
                    "Category": None,
                },
                2: {
                    "Category": "default",
                    "User features": [1.0, 0.0],
                },
                3: {
                    "Category": "default",
                    "User features": [1.0, 1.0],
                },
            }
        }
    }
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
        "default": [
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.1],
            [1.0, 1.0, 0.1],
        ]
    }
    y_expected = {
        "default": [
            [True, False],
            [False, True],
            [True, False],
        ]
    }
    xy = PrimalSolutionComponent.xy_sample(features, sample)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected


def test_xy_sample_without_lp_solution() -> None:
    features: Features = {
        "Variables": {
            "x": {
                0: {
                    "Category": "default",
                    "User features": [0.0, 0.0],
                },
                1: {
                    "Category": None,
                },
                2: {
                    "Category": "default",
                    "User features": [1.0, 0.0],
                },
                3: {
                    "Category": "default",
                    "User features": [1.0, 1.0],
                },
            }
        }
    }
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
        "default": [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    }
    y_expected = {
        "default": [
            [True, False],
            [False, True],
            [True, False],
        ]
    }
    xy = PrimalSolutionComponent.xy_sample(features, sample)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected


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
    instance.features = {
        "Variables": {
            "x": {
                0: {
                    "Category": "default",
                    "User features": [0.0, 0.0],
                },
                1: {
                    "Category": "default",
                    "User features": [0.0, 2.0],
                },
                2: {
                    "Category": "default",
                    "User features": [2.0, 0.0],
                },
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
