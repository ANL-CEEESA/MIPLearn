#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from unittest.mock import Mock, call

from miplearn import (
    StaticLazyConstraintsComponent,
    LearningSolver,
    Instance,
    InternalSolver,
)
from miplearn.classifiers import Classifier


def test_usage_with_solver():
    solver = Mock(spec=LearningSolver)
    solver.use_lazy_cb = False
    solver.gap_tolerance = 1e-4

    internal = solver.internal_solver = Mock(spec=InternalSolver)
    internal.get_constraint_ids = Mock(return_value=["c1", "c2", "c3", "c4"])
    internal.extract_constraint = Mock(side_effect=lambda cid: "<%s>" % cid)
    internal.is_constraint_satisfied = Mock(return_value=False)

    instance = Mock(spec=Instance)
    instance.has_static_lazy_constraints = Mock(return_value=True)
    instance.is_constraint_lazy = Mock(
        side_effect=lambda cid: {
            "c1": False,
            "c2": True,
            "c3": True,
            "c4": True,
        }[cid]
    )
    instance.get_constraint_features = Mock(
        side_effect=lambda cid: {
            "c2": [1.0, 0.0],
            "c3": [0.5, 0.5],
            "c4": [1.0],
        }[cid]
    )
    instance.get_constraint_category = Mock(
        side_effect=lambda cid: {
            "c2": "type-a",
            "c3": "type-a",
            "c4": "type-b",
        }[cid]
    )

    component = StaticLazyConstraintsComponent(
        threshold=0.90, use_two_phase_gap=False, violation_tolerance=1.0
    )
    component.classifiers = {
        "type-a": Mock(spec=Classifier),
        "type-b": Mock(spec=Classifier),
    }
    component.classifiers["type-a"].predict_proba = Mock(
        return_value=[
            [0.20, 0.80],
            [0.05, 0.95],
        ]
    )
    component.classifiers["type-b"].predict_proba = Mock(
        return_value=[
            [0.02, 0.98],
        ]
    )

    # LearningSolver calls before_solve
    component.before_solve(solver, instance, None)

    # Should ask if instance has static lazy constraints
    instance.has_static_lazy_constraints.assert_called_once()

    # Should ask internal solver for a list of constraints in the model
    internal.get_constraint_ids.assert_called_once()

    # Should ask if each constraint in the model is lazy
    instance.is_constraint_lazy.assert_has_calls(
        [
            call("c1"),
            call("c2"),
            call("c3"),
            call("c4"),
        ]
    )

    # For the lazy ones, should ask for features
    instance.get_constraint_features.assert_has_calls(
        [
            call("c2"),
            call("c3"),
            call("c4"),
        ]
    )

    # Should also ask for categories
    assert instance.get_constraint_category.call_count == 3
    instance.get_constraint_category.assert_has_calls(
        [
            call("c2"),
            call("c3"),
            call("c4"),
        ]
    )

    # Should ask internal solver to remove constraints identified as lazy
    assert internal.extract_constraint.call_count == 3
    internal.extract_constraint.assert_has_calls(
        [
            call("c2"),
            call("c3"),
            call("c4"),
        ]
    )

    # Should ask ML to predict whether each lazy constraint should be enforced
    component.classifiers["type-a"].predict_proba.assert_called_once_with(
        [[1.0, 0.0], [0.5, 0.5]]
    )
    component.classifiers["type-b"].predict_proba.assert_called_once_with([[1.0]])

    # For the ones that should be enforced, should ask solver to re-add them
    # to the formulation. The remaining ones should remain in the pool.
    assert internal.add_constraint.call_count == 2
    internal.add_constraint.assert_has_calls(
        [
            call("<c3>"),
            call("<c4>"),
        ]
    )
    internal.add_constraint.reset_mock()

    # LearningSolver calls after_iteration (first time)
    should_repeat = component.iteration_cb(solver, instance, None)
    assert should_repeat

    # Should ask internal solver to verify if constraints in the pool are
    # satisfied and add the ones that are not
    internal.is_constraint_satisfied.assert_called_once_with("<c2>", tol=1.0)
    internal.is_constraint_satisfied.reset_mock()
    internal.add_constraint.assert_called_once_with("<c2>")
    internal.add_constraint.reset_mock()

    # LearningSolver calls after_iteration (second time)
    should_repeat = component.iteration_cb(solver, instance, None)
    assert not should_repeat

    # The lazy constraint pool should be empty by now, so no calls should be made
    internal.is_constraint_satisfied.assert_not_called()
    internal.add_constraint.assert_not_called()

    # Should update instance object
    assert instance.found_violated_lazy_constraints == ["c3", "c4", "c2"]


def test_fit():
    instance_1 = Mock(spec=Instance)
    instance_1.found_violated_lazy_constraints = ["c1", "c2", "c4", "c5"]
    instance_1.get_constraint_category = Mock(
        side_effect=lambda cid: {
            "c1": "type-a",
            "c2": "type-a",
            "c3": "type-a",
            "c4": "type-b",
            "c5": "type-b",
        }[cid]
    )
    instance_1.get_constraint_features = Mock(
        side_effect=lambda cid: {
            "c1": [1, 1],
            "c2": [1, 2],
            "c3": [1, 3],
            "c4": [1, 4, 0],
            "c5": [1, 5, 0],
        }[cid]
    )

    instance_2 = Mock(spec=Instance)
    instance_2.found_violated_lazy_constraints = ["c2", "c3", "c4"]
    instance_2.get_constraint_category = Mock(
        side_effect=lambda cid: {
            "c1": "type-a",
            "c2": "type-a",
            "c3": "type-a",
            "c4": "type-b",
            "c5": "type-b",
        }[cid]
    )
    instance_2.get_constraint_features = Mock(
        side_effect=lambda cid: {
            "c1": [2, 1],
            "c2": [2, 2],
            "c3": [2, 3],
            "c4": [2, 4, 0],
            "c5": [2, 5, 0],
        }[cid]
    )

    instances = [instance_1, instance_2]
    component = StaticLazyConstraintsComponent()
    component.classifiers = {
        "type-a": Mock(spec=Classifier),
        "type-b": Mock(spec=Classifier),
    }

    expected_constraints = {
        "type-a": ["c1", "c2", "c3"],
        "type-b": ["c4", "c5"],
    }
    expected_x = {
        "type-a": [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]],
        "type-b": [[1, 4, 0], [1, 5, 0], [2, 4, 0], [2, 5, 0]],
    }
    expected_y = {
        "type-a": [[0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1]],
        "type-b": [[0, 1], [0, 1], [0, 1], [1, 0]],
    }
    assert component._collect_constraints(instances) == expected_constraints
    assert component.x(instances) == expected_x
    assert component.y(instances) == expected_y

    component.fit(instances)
    component.classifiers["type-a"].fit.assert_called_once_with(
        expected_x["type-a"],
        expected_y["type-a"],
    )
    component.classifiers["type-b"].fit.assert_called_once_with(
        expected_x["type-b"],
        expected_y["type-b"],
    )
