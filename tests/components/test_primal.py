#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal
from scipy.stats import randint

from miplearn import Classifier, LearningSolver, GurobiSolver, GurobiPyomoSolver
from miplearn.classifiers.threshold import Threshold
from miplearn.components.primal import PrimalSolutionComponent
from miplearn.problems.tsp import TravelingSalesmanGenerator
from miplearn.types import TrainingSample, Features
from tests.fixtures.knapsack import get_knapsack_instance


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
    features: Features = {
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
    sample: TrainingSample = {
        "LP solution": {
            "x": {
                0: 0.1,
                1: 0.5,
                2: 0.9,
            }
        }
    }
    x = PrimalSolutionComponent.x_sample(features, sample)
    comp = PrimalSolutionComponent()
    comp.classifiers = {"default": clf}
    comp.thresholds = {"default": thr}
    solution_actual = comp.predict(features, sample)
    clf.predict_proba.assert_called_once()
    assert_array_equal(x["default"], clf.predict_proba.call_args[0][0])
    thr.predict.assert_called_once()
    assert_array_equal(x["default"], thr.predict.call_args[0][0])
    assert solution_actual == {
        "x": {
            0: 0.0,
            1: None,
            2: 1.0,
        }
    }


def test_fit_xy():
    comp = PrimalSolutionComponent(
        classifier=lambda: Mock(spec=Classifier),
        threshold=lambda: Mock(spec=Threshold),
    )
    x = {
        "type-a": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "type-b": np.array([[7.0, 8.0, 9.0]]),
    }
    y = {
        "type-a": np.array([[True, False], [False, True]]),
        "type-b": np.array([[True, False]]),
    }
    comp.fit_xy(x, y)
    for category in ["type-a", "type-b"]:
        assert category in comp.classifiers
        assert category in comp.thresholds
        clf = comp.classifiers[category]
        clf.fit.assert_called_once()
        assert_array_equal(x[category], clf.fit.call_args[0][0])
        assert_array_equal(y[category], clf.fit.call_args[0][1])
        thr = comp.thresholds[category]
        thr.fit.assert_called_once()
        assert_array_equal(x[category], thr.fit.call_args[0][1])
        assert_array_equal(y[category], thr.fit.call_args[0][2])


def test_usage():
    solver = LearningSolver(
        components=[
            PrimalSolutionComponent(),
        ]
    )
    gen = TravelingSalesmanGenerator(n=randint(low=5, high=6))
    instance = gen.generate(1)[0]
    solver.solve(instance)
    solver.fit([instance])
    stats = solver.solve(instance)
    assert stats["Primal: free"] == 0
    assert stats["Primal: one"] + stats["Primal: zero"] == 10
    assert stats["Lower bound"] == stats["Warm start value"]
