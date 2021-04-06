#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal
from scipy.stats import randint

from miplearn.classifiers import Classifier
from miplearn.classifiers.threshold import Threshold
from miplearn.components import classifier_evaluation_dict
from miplearn.components.primal import PrimalSolutionComponent
from miplearn.features import TrainingSample, VariableFeatures, Features
from miplearn.instance import Instance
from miplearn.problems.tsp import TravelingSalesmanGenerator
from miplearn.solvers.learning import LearningSolver


def test_xy() -> None:
    features = Features(
        variables={
            "x": {
                0: VariableFeatures(
                    category="default",
                    user_features=[0.0, 0.0],
                ),
                1: VariableFeatures(
                    category=None,
                ),
                2: VariableFeatures(
                    category="default",
                    user_features=[1.0, 0.0],
                ),
                3: VariableFeatures(
                    category="default",
                    user_features=[1.0, 1.0],
                ),
            }
        }
    )
    instance = Mock(spec=Instance)
    instance.features = features
    sample = TrainingSample(
        solution={
            "x": {
                0: 0.0,
                1: 1.0,
                2: 1.0,
                3: 0.0,
            }
        },
        lp_solution={
            "x": {
                0: 0.1,
                1: 0.1,
                2: 0.1,
                3: 0.1,
            }
        },
    )
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
    xy = PrimalSolutionComponent.sample_xy(instance, sample)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected


def test_xy_without_lp_solution() -> None:
    features = Features(
        variables={
            "x": {
                0: VariableFeatures(
                    category="default",
                    user_features=[0.0, 0.0],
                ),
                1: VariableFeatures(
                    category=None,
                ),
                2: VariableFeatures(
                    category="default",
                    user_features=[1.0, 0.0],
                ),
                3: VariableFeatures(
                    category="default",
                    user_features=[1.0, 1.0],
                ),
            }
        }
    )
    instance = Mock(spec=Instance)
    instance.features = features
    sample = TrainingSample(
        solution={
            "x": {
                0: 0.0,
                1: 1.0,
                2: 1.0,
                3: 0.0,
            }
        },
    )
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
    xy = PrimalSolutionComponent.sample_xy(instance, sample)
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
    features = Features(
        variables={
            "x": {
                0: VariableFeatures(
                    category="default",
                    user_features=[0.0, 0.0],
                ),
                1: VariableFeatures(
                    category="default",
                    user_features=[0.0, 2.0],
                ),
                2: VariableFeatures(
                    category="default",
                    user_features=[2.0, 0.0],
                ),
            }
        }
    )
    instance = Mock(spec=Instance)
    instance.features = features
    sample = TrainingSample(
        lp_solution={
            "x": {
                0: 0.1,
                1: 0.5,
                2: 0.9,
            }
        }
    )
    x, _ = PrimalSolutionComponent.sample_xy(instance, sample)
    comp = PrimalSolutionComponent()
    comp.classifiers = {"default": clf}
    comp.thresholds = {"default": thr}
    solution_actual = comp.sample_predict(instance, sample)
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
    clf = Mock(spec=Classifier)
    clf.clone = lambda: Mock(spec=Classifier)
    thr = Mock(spec=Threshold)
    thr.clone = lambda: Mock(spec=Threshold)
    comp = PrimalSolutionComponent(classifier=clf, threshold=thr)
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
    assert stats["Primal: Free"] == 0
    assert stats["Primal: One"] + stats["Primal: Zero"] == 10
    assert stats["Lower bound"] == stats["Warm start value"]


def test_evaluate() -> None:
    comp = PrimalSolutionComponent()
    comp.sample_predict = lambda _, __: {  # type: ignore
        "x": {
            0: 1.0,
            1: 0.0,
            2: 0.0,
            3: None,
            4: 1.0,
        }
    }
    features: Features = Features(
        variables={
            "x": {
                0: VariableFeatures(),
                1: VariableFeatures(),
                2: VariableFeatures(),
                3: VariableFeatures(),
                4: VariableFeatures(),
            }
        }
    )
    instance = Mock(spec=Instance)
    instance.features = features
    sample: TrainingSample = TrainingSample(
        solution={
            "x": {
                0: 1.0,
                1: 1.0,
                2: 0.0,
                3: 1.0,
                4: 1.0,
            }
        }
    )
    ev = comp.sample_evaluate(instance, sample)
    assert ev == {
        0: classifier_evaluation_dict(tp=1, fp=1, tn=3, fn=0),
        1: classifier_evaluation_dict(tp=2, fp=0, tn=1, fn=2),
    }
