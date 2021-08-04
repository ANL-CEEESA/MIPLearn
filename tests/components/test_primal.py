#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.stats import randint

from miplearn.classifiers import Classifier
from miplearn.classifiers.threshold import Threshold
from miplearn.components import classifier_evaluation_dict
from miplearn.components.primal import PrimalSolutionComponent
from miplearn.features.sample import Sample, MemorySample
from miplearn.problems.tsp import TravelingSalesmanGenerator
from miplearn.solvers.learning import LearningSolver
from miplearn.solvers.tests import assert_equals


@pytest.fixture
def sample() -> Sample:
    sample = MemorySample(
        {
            "static_var_names": ["x[0]", "x[1]", "x[2]", "x[3]"],
            "static_var_categories": ["default", None, "default", "default"],
            "mip_var_values": [0.0, 1.0, 1.0, 0.0],
            "static_instance_features": [5.0],
            "static_var_features": [
                [0.0, 0.0],
                None,
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            "lp_var_features": [
                [0.0, 0.0, 2.0, 2.0],
                None,
                [1.0, 0.0, 3.0, 2.0],
                [1.0, 1.0, 3.0, 3.0],
            ],
        },
    )
    return sample


def test_xy(sample: Sample) -> None:
    x_expected = {
        "default": [
            [5.0, 0.0, 0.0, 2.0, 2.0],
            [5.0, 1.0, 0.0, 3.0, 2.0],
            [5.0, 1.0, 1.0, 3.0, 3.0],
        ]
    }
    y_expected = {
        "default": [
            [True, False],
            [False, True],
            [True, False],
        ]
    }
    xy = PrimalSolutionComponent().sample_xy(None, sample)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected


def test_fit_xy() -> None:
    clf = Mock(spec=Classifier)
    clf.clone = lambda: Mock(spec=Classifier)  # type: ignore
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
        clf = comp.classifiers[category]  # type: ignore
        clf.fit.assert_called_once()
        assert_array_equal(x[category], clf.fit.call_args[0][0])
        assert_array_equal(y[category], clf.fit.call_args[0][1])
        thr = comp.thresholds[category]  # type: ignore
        thr.fit.assert_called_once()
        assert_array_equal(x[category], thr.fit.call_args[0][1])
        assert_array_equal(y[category], thr.fit.call_args[0][2])


def test_usage() -> None:
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
    assert stats["mip_lower_bound"] == stats["mip_warm_start_value"]


def test_evaluate(sample: Sample) -> None:
    comp = PrimalSolutionComponent()
    comp.sample_predict = lambda _: {  # type: ignore
        "x[0]": 1.0,
        "x[1]": 1.0,
        "x[2]": 0.0,
        "x[3]": None,
    }
    ev = comp.sample_evaluate(None, sample)
    assert_equals(
        ev,
        {
            "0": classifier_evaluation_dict(tp=0, fp=1, tn=1, fn=2),
            "1": classifier_evaluation_dict(tp=1, fp=1, tn=1, fn=1),
        },
    )


def test_predict(sample: Sample) -> None:
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
    comp = PrimalSolutionComponent()
    x, _ = comp.sample_xy(None, sample)
    comp.classifiers = {"default": clf}
    comp.thresholds = {"default": thr}
    pred = comp.sample_predict(sample)
    clf.predict_proba.assert_called_once()
    thr.predict.assert_called_once()
    assert_array_equal(x["default"], clf.predict_proba.call_args[0][0])
    assert_array_equal(x["default"], thr.predict.call_args[0][0])
    assert pred == {
        "x[0]": 0.0,
        "x[1]": None,
        "x[2]": None,
        "x[3]": 1.0,
    }
