#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Dict, cast, Hashable
from unittest.mock import Mock, call

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from miplearn import LearningSolver, InternalSolver, Instance
from miplearn.classifiers import Classifier
from miplearn.classifiers.threshold import Threshold, MinProbabilityThreshold
from miplearn.components.lazy_static import StaticLazyConstraintsComponent
from miplearn.types import TrainingSample, Features, LearningSolveStats


@pytest.fixture
def sample() -> TrainingSample:
    return {
        "LazyStatic: Enforced": {"c1", "c2", "c4"},
    }


@pytest.fixture
def features() -> Features:
    return {
        "Instance": {
            "Lazy constraint count": 4,
        },
        "Constraints": {
            "c1": {
                "Category": "type-a",
                "User features": [1.0, 1.0],
                "Lazy": True,
            },
            "c2": {
                "Category": "type-a",
                "User features": [1.0, 2.0],
                "Lazy": True,
            },
            "c3": {
                "Category": "type-a",
                "User features": [1.0, 3.0],
                "Lazy": True,
            },
            "c4": {
                "Category": "type-b",
                "User features": [1.0, 4.0, 0.0],
                "Lazy": True,
            },
            "c5": {
                "Category": "type-b",
                "User features": [1.0, 5.0, 0.0],
                "Lazy": False,
            },
        },
    }


def test_usage_with_solver(features: Features) -> None:
    solver = Mock(spec=LearningSolver)
    solver.use_lazy_cb = False
    solver.gap_tolerance = 1e-4

    internal = solver.internal_solver = Mock(spec=InternalSolver)
    internal.extract_constraint = Mock(side_effect=lambda cid: "<%s>" % cid)
    internal.is_constraint_satisfied = Mock(return_value=False)

    instance = Mock(spec=Instance)
    instance.has_static_lazy_constraints = Mock(return_value=True)

    component = StaticLazyConstraintsComponent(violation_tolerance=1.0)
    component.thresholds["type-a"] = MinProbabilityThreshold([0.5, 0.5])
    component.thresholds["type-b"] = MinProbabilityThreshold([0.5, 0.5])
    component.classifiers = {
        "type-a": Mock(spec=Classifier),
        "type-b": Mock(spec=Classifier),
    }
    component.classifiers["type-a"].predict_proba = Mock(  # type: ignore
        return_value=np.array(
            [
                [0.00, 1.00],  # c1
                [0.20, 0.80],  # c2
                [0.99, 0.01],  # c3
            ]
        )
    )
    component.classifiers["type-b"].predict_proba = Mock(  # type: ignore
        return_value=np.array(
            [
                [0.02, 0.98],  # c4
            ]
        )
    )

    sample: TrainingSample = {}
    stats: LearningSolveStats = {}

    # LearningSolver calls before_solve_mip
    component.before_solve_mip(
        solver=solver,
        instance=instance,
        model=None,
        stats=stats,
        features=features,
        training_data=sample,
    )

    # Should ask ML to predict whether each lazy constraint should be enforced
    component.classifiers["type-a"].predict_proba.assert_called_once()
    component.classifiers["type-b"].predict_proba.assert_called_once()

    # Should ask internal solver to remove some constraints
    assert internal.extract_constraint.call_count == 1
    internal.extract_constraint.assert_has_calls([call("c3")])

    # LearningSolver calls after_iteration (first time)
    should_repeat = component.iteration_cb(solver, instance, None)
    assert should_repeat

    # Should ask internal solver to verify if constraints in the pool are
    # satisfied and add the ones that are not
    internal.is_constraint_satisfied.assert_called_once_with("<c3>", tol=1.0)
    internal.is_constraint_satisfied.reset_mock()
    internal.add_constraint.assert_called_once_with("<c3>")
    internal.add_constraint.reset_mock()

    # LearningSolver calls after_iteration (second time)
    should_repeat = component.iteration_cb(solver, instance, None)
    assert not should_repeat

    # The lazy constraint pool should be empty by now, so no calls should be made
    internal.is_constraint_satisfied.assert_not_called()
    internal.add_constraint.assert_not_called()

    # LearningSolver calls after_solve_mip
    component.after_solve_mip(
        solver=solver,
        instance=instance,
        model=None,
        stats=stats,
        features=features,
        training_data=sample,
    )

    # Should update training sample
    assert sample["LazyStatic: Enforced"] == {"c1", "c2", "c3", "c4"}

    # Should update stats
    assert stats["LazyStatic: Removed"] == 1
    assert stats["LazyStatic: Kept"] == 3
    assert stats["LazyStatic: Restored"] == 1
    assert stats["LazyStatic: Iterations"] == 1


def test_sample_predict(
    features: Features,
    sample: TrainingSample,
) -> None:
    comp = StaticLazyConstraintsComponent()
    comp.thresholds["type-a"] = MinProbabilityThreshold([0.5, 0.5])
    comp.thresholds["type-b"] = MinProbabilityThreshold([0.5, 0.5])
    comp.classifiers["type-a"] = Mock(spec=Classifier)
    comp.classifiers["type-a"].predict_proba = lambda _: np.array(  # type:ignore
        [
            [0.0, 1.0],  # c1
            [0.0, 0.9],  # c2
            [0.9, 0.1],  # c3
        ]
    )
    comp.classifiers["type-b"] = Mock(spec=Classifier)
    comp.classifiers["type-b"].predict_proba = lambda _: np.array(  # type:ignore
        [
            [0.0, 1.0],  # c4
        ]
    )
    pred = comp.sample_predict(features, sample)
    assert pred == ["c1", "c2", "c4"]


def test_fit_xy() -> None:
    x = cast(
        Dict[Hashable, np.ndarray],
        {
            "type-a": np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
            "type-b": np.array([[1.0, 4.0, 0.0]]),
        },
    )
    y = cast(
        Dict[Hashable, np.ndarray],
        {
            "type-a": np.array([[False, True], [False, True], [True, False]]),
            "type-b": np.array([[False, True]]),
        },
    )
    clf: Classifier = Mock(spec=Classifier)
    thr: Threshold = Mock(spec=Threshold)
    clf.clone = Mock(side_effect=lambda: Mock(spec=Classifier))  # type: ignore
    thr.clone = Mock(side_effect=lambda: Mock(spec=Threshold))  # type: ignore
    comp = StaticLazyConstraintsComponent(
        classifier=clf,
        threshold=thr,
    )
    comp.fit_xy(x, y)
    assert clf.clone.call_count == 2
    clf_a = comp.classifiers["type-a"]
    clf_b = comp.classifiers["type-b"]
    assert clf_a.fit.call_count == 1  # type: ignore
    assert clf_b.fit.call_count == 1  # type: ignore
    assert_array_equal(clf_a.fit.call_args[0][0], x["type-a"])  # type: ignore
    assert_array_equal(clf_b.fit.call_args[0][0], x["type-b"])  # type: ignore
    assert thr.clone.call_count == 2
    thr_a = comp.thresholds["type-a"]
    thr_b = comp.thresholds["type-b"]
    assert thr_a.fit.call_count == 1  # type: ignore
    assert thr_b.fit.call_count == 1  # type: ignore
    assert thr_a.fit.call_args[0][0] == clf_a  # type: ignore
    assert thr_b.fit.call_args[0][0] == clf_b  # type: ignore


def test_sample_xy(
    features: Features,
    sample: TrainingSample,
) -> None:
    x_expected = {
        "type-a": [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
        "type-b": [[1.0, 4.0, 0.0]],
    }
    y_expected = {
        "type-a": [[False, True], [False, True], [True, False]],
        "type-b": [[False, True]],
    }
    xy = StaticLazyConstraintsComponent.sample_xy(features, sample)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected
