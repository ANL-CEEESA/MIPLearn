#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from unittest.mock import Mock

import pytest
from numpy.testing import assert_array_equal

from miplearn import GurobiPyomoSolver, LearningSolver, Regressor
from miplearn.components.objective import ObjectiveValueComponent
from miplearn.types import TrainingSample, Features
from tests.fixtures.knapsack import get_knapsack_instance

import numpy as np


@pytest.fixture
def features() -> Features:
    return {
        "Instance": {
            "User features": [1.0, 2.0],
        }
    }


@pytest.fixture
def sample() -> TrainingSample:
    return {
        "Lower bound": 1.0,
        "Upper bound": 2.0,
        "LP value": 3.0,
    }


@pytest.fixture
def sample_without_lp() -> TrainingSample:
    return {
        "Lower bound": 1.0,
        "Upper bound": 2.0,
    }


@pytest.fixture
def sample_without_ub() -> TrainingSample:
    return {
        "Lower bound": 1.0,
        "LP value": 3.0,
    }


def test_sample_xy(
    features: Features,
    sample: TrainingSample,
) -> None:
    x_expected = {
        "Lower bound": [[1.0, 2.0, 3.0]],
        "Upper bound": [[1.0, 2.0, 3.0]],
    }
    y_expected = {
        "Lower bound": [[1.0]],
        "Upper bound": [[2.0]],
    }
    xy = ObjectiveValueComponent.sample_xy(features, sample)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected


def test_sample_xy_without_lp(
    features: Features,
    sample_without_lp: TrainingSample,
) -> None:
    x_expected = {
        "Lower bound": [[1.0, 2.0]],
        "Upper bound": [[1.0, 2.0]],
    }
    y_expected = {
        "Lower bound": [[1.0]],
        "Upper bound": [[2.0]],
    }
    xy = ObjectiveValueComponent.sample_xy(features, sample_without_lp)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected


def test_sample_xy_without_ub(
    features: Features,
    sample_without_ub: TrainingSample,
) -> None:
    x_expected = {
        "Lower bound": [[1.0, 2.0, 3.0]],
        "Upper bound": [[1.0, 2.0, 3.0]],
    }
    y_expected = {"Lower bound": [[1.0]]}
    xy = ObjectiveValueComponent.sample_xy(features, sample_without_ub)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected


def test_fit_xy() -> None:
    x = {
        "Lower bound": np.array([[0.0, 0.0], [1.0, 2.0]]),
        "Upper bound": np.array([[0.0, 0.0], [1.0, 2.0]]),
    }
    y = {
        "Lower bound": np.array([[100.0]]),
        "Upper bound": np.array([[200.0]]),
    }
    reg = Mock(spec=Regressor)
    reg.clone = Mock(side_effect=lambda: Mock(spec=Regressor))
    comp = ObjectiveValueComponent(regressor=reg)
    assert comp.ub_regressor is None
    assert comp.lb_regressor is None
    comp.fit_xy(x, y)
    assert reg.clone.call_count == 2
    assert comp.ub_regressor is not None
    assert comp.lb_regressor is not None
    assert comp.ub_regressor.fit.call_count == 1
    assert comp.lb_regressor.fit.call_count == 1
    assert_array_equal(comp.ub_regressor.fit.call_args[0][0], x["Upper bound"])
    assert_array_equal(comp.lb_regressor.fit.call_args[0][0], x["Lower bound"])
    assert_array_equal(comp.ub_regressor.fit.call_args[0][1], y["Upper bound"])
    assert_array_equal(comp.lb_regressor.fit.call_args[0][1], y["Lower bound"])


def test_fit_xy_without_ub() -> None:
    x = {
        "Lower bound": np.array([[0.0, 0.0], [1.0, 2.0]]),
        "Upper bound": np.array([[0.0, 0.0], [1.0, 2.0]]),
    }
    y = {
        "Lower bound": np.array([[100.0]]),
    }
    reg = Mock(spec=Regressor)
    reg.clone = Mock(side_effect=lambda: Mock(spec=Regressor))
    comp = ObjectiveValueComponent(regressor=reg)
    assert comp.ub_regressor is None
    assert comp.lb_regressor is None
    comp.fit_xy(x, y)
    assert reg.clone.call_count == 1
    assert comp.ub_regressor is None
    assert comp.lb_regressor is not None
    assert comp.lb_regressor.fit.call_count == 1
    assert_array_equal(comp.lb_regressor.fit.call_args[0][0], x["Lower bound"])
    assert_array_equal(comp.lb_regressor.fit.call_args[0][1], y["Lower bound"])


def test_sample_predict(
    features: Features,
    sample: TrainingSample,
) -> None:
    x, y = ObjectiveValueComponent.sample_xy(features, sample)
    comp = ObjectiveValueComponent()
    comp.lb_regressor = Mock(spec=Regressor)
    comp.ub_regressor = Mock(spec=Regressor)
    comp.lb_regressor.predict = Mock(side_effect=lambda _: np.array([[50.0]]))
    comp.ub_regressor.predict = Mock(side_effect=lambda _: np.array([[60.0]]))
    pred = comp.sample_predict(features, sample)
    assert pred == {
        "Lower bound": 50.0,
        "Upper bound": 60.0,
    }
    assert_array_equal(comp.ub_regressor.predict.call_args[0][0], x["Upper bound"])
    assert_array_equal(comp.lb_regressor.predict.call_args[0][0], x["Lower bound"])


def test_sample_predict_without_ub(
    features: Features,
    sample_without_ub: TrainingSample,
) -> None:
    x, y = ObjectiveValueComponent.sample_xy(features, sample_without_ub)
    comp = ObjectiveValueComponent()
    comp.lb_regressor = Mock(spec=Regressor)
    comp.lb_regressor.predict = Mock(side_effect=lambda _: np.array([[50.0]]))
    pred = comp.sample_predict(features, sample_without_ub)
    assert pred == {
        "Lower bound": 50.0,
    }
    assert_array_equal(comp.lb_regressor.predict.call_args[0][0], x["Lower bound"])


def test_sample_evaluate(features: Features, sample: TrainingSample) -> None:
    comp = ObjectiveValueComponent()
    comp.lb_regressor = Mock(spec=Regressor)
    comp.lb_regressor.predict = lambda _: np.array([[1.05]])
    comp.ub_regressor = Mock(spec=Regressor)
    comp.ub_regressor.predict = lambda _: np.array([[2.50]])
    ev = comp.sample_evaluate(features, sample)
    assert ev == {
        "Lower bound": {
            "Actual value": 1.0,
            "Predicted value": 1.05,
            "Absolute error": 0.05,
            "Relative error": 0.05,
        },
        "Upper bound": {
            "Actual value": 2.0,
            "Predicted value": 2.50,
            "Absolute error": 0.5,
            "Relative error": 0.25,
        },
    }


def test_usage() -> None:
    solver = LearningSolver(components=[ObjectiveValueComponent()])
    instance = get_knapsack_instance(GurobiPyomoSolver())
    solver.solve(instance)
    solver.fit([instance])
    stats = solver.solve(instance)
    assert stats["Lower bound"] == stats["Objective: Predicted LB"]
    assert stats["Upper bound"] == stats["Objective: Predicted UB"]
