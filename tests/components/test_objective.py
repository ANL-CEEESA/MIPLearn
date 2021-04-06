#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Hashable, Dict
from unittest.mock import Mock

import pytest
from numpy.testing import assert_array_equal

from miplearn.classifiers import Regressor
from miplearn.components.objective import ObjectiveValueComponent
from miplearn.features import TrainingSample, InstanceFeatures, Features
from miplearn.instance import Instance
from miplearn.solvers.learning import LearningSolver
from miplearn.solvers.pyomo.gurobi import GurobiPyomoSolver
from tests.fixtures.knapsack import get_knapsack_instance

import numpy as np


@pytest.fixture
def instance(features: Features) -> Instance:
    instance = Mock(spec=Instance)
    instance.features = features
    return instance


@pytest.fixture
def features() -> Features:
    return Features(
        instance=InstanceFeatures(
            user_features=[1.0, 2.0],
        )
    )


@pytest.fixture
def sample() -> TrainingSample:
    return TrainingSample(
        lower_bound=1.0,
        upper_bound=2.0,
        lp_value=3.0,
    )


@pytest.fixture
def sample_without_lp() -> TrainingSample:
    return TrainingSample(
        lower_bound=1.0,
        upper_bound=2.0,
    )


@pytest.fixture
def sample_without_ub() -> TrainingSample:
    return TrainingSample(
        lower_bound=1.0,
        lp_value=3.0,
    )


def test_sample_xy(
    instance: Instance,
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
    xy = ObjectiveValueComponent().sample_xy(instance, sample)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected


def test_sample_xy_without_lp(
    instance: Instance,
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
    xy = ObjectiveValueComponent().sample_xy(instance, sample_without_lp)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected


def test_sample_xy_without_ub(
    instance: Instance,
    sample_without_ub: TrainingSample,
) -> None:
    x_expected = {
        "Lower bound": [[1.0, 2.0, 3.0]],
        "Upper bound": [[1.0, 2.0, 3.0]],
    }
    y_expected = {"Lower bound": [[1.0]]}
    xy = ObjectiveValueComponent().sample_xy(instance, sample_without_ub)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected


def test_fit_xy() -> None:
    x: Dict[Hashable, np.ndarray] = {
        "Lower bound": np.array([[0.0, 0.0], [1.0, 2.0]]),
        "Upper bound": np.array([[0.0, 0.0], [1.0, 2.0]]),
    }
    y: Dict[Hashable, np.ndarray] = {
        "Lower bound": np.array([[100.0]]),
        "Upper bound": np.array([[200.0]]),
    }
    reg = Mock(spec=Regressor)
    reg.clone = Mock(side_effect=lambda: Mock(spec=Regressor))
    comp = ObjectiveValueComponent(regressor=reg)
    assert "Upper bound" not in comp.regressors
    assert "Lower bound" not in comp.regressors
    comp.fit_xy(x, y)
    assert reg.clone.call_count == 2
    assert "Upper bound" in comp.regressors
    assert "Lower bound" in comp.regressors
    assert comp.regressors["Upper bound"].fit.call_count == 1  # type: ignore
    assert comp.regressors["Lower bound"].fit.call_count == 1  # type: ignore
    assert_array_equal(
        comp.regressors["Upper bound"].fit.call_args[0][0],  # type: ignore
        x["Upper bound"],
    )
    assert_array_equal(
        comp.regressors["Lower bound"].fit.call_args[0][0],  # type: ignore
        x["Lower bound"],
    )
    assert_array_equal(
        comp.regressors["Upper bound"].fit.call_args[0][1],  # type: ignore
        y["Upper bound"],
    )
    assert_array_equal(
        comp.regressors["Lower bound"].fit.call_args[0][1],  # type: ignore
        y["Lower bound"],
    )


def test_fit_xy_without_ub() -> None:
    x: Dict[Hashable, np.ndarray] = {
        "Lower bound": np.array([[0.0, 0.0], [1.0, 2.0]]),
        "Upper bound": np.array([[0.0, 0.0], [1.0, 2.0]]),
    }
    y: Dict[Hashable, np.ndarray] = {
        "Lower bound": np.array([[100.0]]),
    }
    reg = Mock(spec=Regressor)
    reg.clone = Mock(side_effect=lambda: Mock(spec=Regressor))
    comp = ObjectiveValueComponent(regressor=reg)
    assert "Upper bound" not in comp.regressors
    assert "Lower bound" not in comp.regressors
    comp.fit_xy(x, y)
    assert reg.clone.call_count == 1
    assert "Upper bound" not in comp.regressors
    assert "Lower bound" in comp.regressors
    assert comp.regressors["Lower bound"].fit.call_count == 1  # type: ignore
    assert_array_equal(
        comp.regressors["Lower bound"].fit.call_args[0][0],  # type: ignore
        x["Lower bound"],
    )
    assert_array_equal(
        comp.regressors["Lower bound"].fit.call_args[0][1],  # type: ignore
        y["Lower bound"],
    )


def test_sample_predict(
    instance: Instance,
    sample: TrainingSample,
) -> None:
    x, y = ObjectiveValueComponent().sample_xy(instance, sample)
    comp = ObjectiveValueComponent()
    comp.regressors["Lower bound"] = Mock(spec=Regressor)
    comp.regressors["Upper bound"] = Mock(spec=Regressor)
    comp.regressors["Lower bound"].predict = Mock(  # type: ignore
        side_effect=lambda _: np.array([[50.0]])
    )
    comp.regressors["Upper bound"].predict = Mock(  # type: ignore
        side_effect=lambda _: np.array([[60.0]])
    )
    pred = comp.sample_predict(instance, sample)
    assert pred == {
        "Lower bound": 50.0,
        "Upper bound": 60.0,
    }
    assert_array_equal(
        comp.regressors["Upper bound"].predict.call_args[0][0],  # type: ignore
        x["Upper bound"],
    )
    assert_array_equal(
        comp.regressors["Lower bound"].predict.call_args[0][0],  # type: ignore
        x["Lower bound"],
    )


def test_sample_predict_without_ub(
    instance: Instance,
    sample_without_ub: TrainingSample,
) -> None:
    x, y = ObjectiveValueComponent().sample_xy(instance, sample_without_ub)
    comp = ObjectiveValueComponent()
    comp.regressors["Lower bound"] = Mock(spec=Regressor)
    comp.regressors["Lower bound"].predict = Mock(  # type: ignore
        side_effect=lambda _: np.array([[50.0]])
    )
    pred = comp.sample_predict(instance, sample_without_ub)
    assert pred == {
        "Lower bound": 50.0,
    }
    assert_array_equal(
        comp.regressors["Lower bound"].predict.call_args[0][0],  # type: ignore
        x["Lower bound"],
    )


def test_sample_evaluate(instance: Instance, sample: TrainingSample) -> None:
    comp = ObjectiveValueComponent()
    comp.regressors["Lower bound"] = Mock(spec=Regressor)
    comp.regressors["Lower bound"].predict = lambda _: np.array([[1.05]])  # type: ignore
    comp.regressors["Upper bound"] = Mock(spec=Regressor)
    comp.regressors["Upper bound"].predict = lambda _: np.array([[2.50]])  # type: ignore
    ev = comp.sample_evaluate(instance, sample)
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
    assert stats["Lower bound"] == stats["Objective: Predicted lower bound"]
    assert stats["Upper bound"] == stats["Objective: Predicted upper bound"]
