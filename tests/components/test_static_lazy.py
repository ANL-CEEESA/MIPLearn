#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Dict, cast
from unittest.mock import Mock, call

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from miplearn.classifiers import Classifier
from miplearn.classifiers.threshold import Threshold, MinProbabilityThreshold
from miplearn.components.static_lazy import StaticLazyConstraintsComponent
from miplearn.features.sample import Sample, MemorySample
from miplearn.instance.base import Instance
from miplearn.solvers.internal import InternalSolver, Constraints
from miplearn.solvers.learning import LearningSolver
from miplearn.types import (
    LearningSolveStats,
    ConstraintCategory,
)
from miplearn.solvers.tests import assert_equals


@pytest.fixture
def sample() -> Sample:
    sample = MemorySample(
        {
            "static_constr_categories": [
                b"type-a",
                b"type-a",
                b"type-a",
                b"type-b",
                b"type-b",
            ],
            "static_constr_lazy": np.array([True, True, True, True, False]),
            "static_constr_names": np.array(["c1", "c2", "c3", "c4", "c5"], dtype="S"),
            "static_instance_features": [5.0],
            "mip_constr_lazy_enforced": np.array(["c1", "c2", "c4"], dtype="S"),
            "lp_constr_features": np.array(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [1.0, 3.0, 0.0],
                    [1.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "static_constr_lazy_count": 4,
        },
    )
    return sample


@pytest.fixture
def instance(sample: Sample) -> Instance:
    instance = Mock(spec=Instance)
    instance.get_samples = Mock(return_value=[sample])  # type: ignore
    instance.has_static_lazy_constraints = Mock(return_value=True)
    return instance


def test_usage_with_solver(instance: Instance) -> None:
    solver = Mock(spec=LearningSolver)
    solver.use_lazy_cb = False
    solver.gap_tolerance = 1e-4

    internal = solver.internal_solver = Mock(spec=InternalSolver)
    internal.is_constraint_satisfied_old = Mock(return_value=False)
    internal.are_constraints_satisfied = Mock(
        side_effect=lambda cf, tol=1.0: [False for i in range(len(cf.names))]
    )

    component = StaticLazyConstraintsComponent(violation_tolerance=1.0)
    component.thresholds[b"type-a"] = MinProbabilityThreshold([0.5, 0.5])
    component.thresholds[b"type-b"] = MinProbabilityThreshold([0.5, 0.5])
    component.classifiers = {
        b"type-a": Mock(spec=Classifier),
        b"type-b": Mock(spec=Classifier),
    }
    component.classifiers[b"type-a"].predict_proba = Mock(  # type: ignore
        return_value=np.array(
            [
                [0.00, 1.00],  # c1
                [0.20, 0.80],  # c2
                [0.99, 0.01],  # c3
            ]
        )
    )
    component.classifiers[b"type-b"].predict_proba = Mock(  # type: ignore
        return_value=np.array(
            [
                [0.02, 0.98],  # c4
            ]
        )
    )

    stats: LearningSolveStats = {}
    sample = instance.get_samples()[0]
    assert sample.get_array("mip_constr_lazy_enforced") is not None

    # LearningSolver calls before_solve_mip
    component.before_solve_mip(
        solver=solver,
        instance=instance,
        model=None,
        stats=stats,
        sample=sample,
    )

    # Should ask ML to predict whether each lazy constraint should be enforced
    component.classifiers[b"type-a"].predict_proba.assert_called_once()
    component.classifiers[b"type-b"].predict_proba.assert_called_once()

    # Should ask internal solver to remove some constraints
    assert internal.remove_constraints.call_count == 1
    internal.remove_constraints.assert_has_calls([call([b"c3"])])

    # LearningSolver calls after_iteration (first time)
    should_repeat = component.iteration_cb(solver, instance, None)
    assert should_repeat

    # Should ask internal solver to verify if constraints in the pool are
    # satisfied and add the ones that are not
    c = Constraints.from_sample(sample)[[False, False, True, False, False]]
    internal.are_constraints_satisfied.assert_called_once_with(c, tol=1.0)
    internal.are_constraints_satisfied.reset_mock()
    internal.add_constraints.assert_called_once_with(c)
    internal.add_constraints.reset_mock()

    # LearningSolver calls after_iteration (second time)
    should_repeat = component.iteration_cb(solver, instance, None)
    assert not should_repeat

    # The lazy constraint pool should be empty by now, so no calls should be made
    internal.are_constraints_satisfied.assert_not_called()
    internal.add_constraints.assert_not_called()

    # LearningSolver calls after_solve_mip
    component.after_solve_mip(
        solver=solver,
        instance=instance,
        model=None,
        stats=stats,
        sample=sample,
    )

    # Should update training sample
    mip_constr_lazy_enforced = sample.get_array("mip_constr_lazy_enforced")
    assert mip_constr_lazy_enforced is not None
    assert_equals(
        sorted(mip_constr_lazy_enforced),
        np.array(["c1", "c2", "c3", "c4"], dtype="S"),
    )

    # Should update stats
    assert stats["LazyStatic: Removed"] == 1
    assert stats["LazyStatic: Kept"] == 3
    assert stats["LazyStatic: Restored"] == 1
    assert stats["LazyStatic: Iterations"] == 1


def test_sample_predict(sample: Sample) -> None:
    comp = StaticLazyConstraintsComponent()
    comp.thresholds[b"type-a"] = MinProbabilityThreshold([0.5, 0.5])
    comp.thresholds[b"type-b"] = MinProbabilityThreshold([0.5, 0.5])
    comp.classifiers[b"type-a"] = Mock(spec=Classifier)
    comp.classifiers[b"type-a"].predict_proba = lambda _: np.array(  # type:ignore
        [
            [0.0, 1.0],  # c1
            [0.0, 0.9],  # c2
            [0.9, 0.1],  # c3
        ]
    )
    comp.classifiers[b"type-b"] = Mock(spec=Classifier)
    comp.classifiers[b"type-b"].predict_proba = lambda _: np.array(  # type:ignore
        [
            [0.0, 1.0],  # c4
        ]
    )
    pred = comp.sample_predict(sample)
    assert pred == [b"c1", b"c2", b"c4"]


def test_fit_xy() -> None:
    x = cast(
        Dict[ConstraintCategory, np.ndarray],
        {
            b"type-a": np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
            b"type-b": np.array([[1.0, 4.0, 0.0]]),
        },
    )
    y = cast(
        Dict[ConstraintCategory, np.ndarray],
        {
            b"type-a": np.array([[False, True], [False, True], [True, False]]),
            b"type-b": np.array([[False, True]]),
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
    clf_a = comp.classifiers[b"type-a"]
    clf_b = comp.classifiers[b"type-b"]
    assert clf_a.fit.call_count == 1  # type: ignore
    assert clf_b.fit.call_count == 1  # type: ignore
    assert_array_equal(clf_a.fit.call_args[0][0], x[b"type-a"])  # type: ignore
    assert_array_equal(clf_b.fit.call_args[0][0], x[b"type-b"])  # type: ignore
    assert thr.clone.call_count == 2
    thr_a = comp.thresholds[b"type-a"]
    thr_b = comp.thresholds[b"type-b"]
    assert thr_a.fit.call_count == 1  # type: ignore
    assert thr_b.fit.call_count == 1  # type: ignore
    assert thr_a.fit.call_args[0][0] == clf_a  # type: ignore
    assert thr_b.fit.call_args[0][0] == clf_b  # type: ignore


def test_sample_xy(sample: Sample) -> None:
    x_expected = {
        b"type-a": [[5.0, 1.0, 1.0, 0.0], [5.0, 1.0, 2.0, 0.0], [5.0, 1.0, 3.0, 0.0]],
        b"type-b": [[5.0, 1.0, 4.0, 0.0]],
    }
    y_expected = {
        b"type-a": [[False, True], [False, True], [True, False]],
        b"type-b": [[False, True]],
    }
    xy = StaticLazyConstraintsComponent().sample_xy(None, sample)
    assert xy is not None
    x_actual, y_actual = xy
    assert x_actual == x_expected
    assert y_actual == y_expected
