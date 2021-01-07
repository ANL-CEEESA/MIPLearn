#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from unittest.mock import Mock, call

from miplearn import LearningSolver, Instance, InternalSolver
from miplearn.classifiers import Classifier
from miplearn.components.relaxation import DropRedundantInequalitiesStep


def _setup():
    solver = Mock(spec=LearningSolver)

    internal = solver.internal_solver = Mock(spec=InternalSolver)
    internal.get_constraint_ids = Mock(return_value=["c1", "c2", "c3", "c4"])
    internal.get_inequality_slacks = Mock(
        side_effect=lambda: {
            "c1": 0.5,
            "c2": 0.0,
            "c3": 0.0,
            "c4": 1.4,
        }
    )
    internal.extract_constraint = Mock(side_effect=lambda cid: "<%s>" % cid)
    internal.is_constraint_satisfied = Mock(return_value=False)

    instance = Mock(spec=Instance)
    instance.get_constraint_features = Mock(
        side_effect=lambda cid: {
            "c2": [1.0, 0.0],
            "c3": [0.5, 0.5],
            "c4": [1.0],
        }[cid]
    )
    instance.get_constraint_category = Mock(
        side_effect=lambda cid: {
            "c1": None,
            "c2": "type-a",
            "c3": "type-a",
            "c4": "type-b",
        }[cid]
    )

    classifiers = {
        "type-a": Mock(spec=Classifier),
        "type-b": Mock(spec=Classifier),
    }
    classifiers["type-a"].predict_proba = Mock(
        return_value=[
            [0.20, 0.80],
            [0.05, 0.95],
        ]
    )
    classifiers["type-b"].predict_proba = Mock(
        return_value=[
            [0.02, 0.98],
        ]
    )

    return solver, internal, instance, classifiers


def test_drop_redundant():
    solver, internal, instance, classifiers = _setup()

    component = DropRedundantInequalitiesStep()
    component.classifiers = classifiers

    # LearningSolver calls before_solve
    component.before_solve(solver, instance, None)

    # Should query list of constraints
    internal.get_constraint_ids.assert_called_once()

    # Should query category and features for each constraint in the model
    assert instance.get_constraint_category.call_count == 4
    instance.get_constraint_category.assert_has_calls(
        [
            call("c1"),
            call("c2"),
            call("c3"),
            call("c4"),
        ]
    )

    # For constraint with non-null categories, should ask for features
    assert instance.get_constraint_features.call_count == 3
    instance.get_constraint_features.assert_has_calls(
        [
            call("c2"),
            call("c3"),
            call("c4"),
        ]
    )

    # Should ask ML to predict whether constraint should be removed
    component.classifiers["type-a"].predict_proba.assert_called_once_with(
        [[1.0, 0.0], [0.5, 0.5]]
    )
    component.classifiers["type-b"].predict_proba.assert_called_once_with([[1.0]])

    # Should ask internal solver to remove constraints predicted as redundant
    assert internal.extract_constraint.call_count == 2
    internal.extract_constraint.assert_has_calls(
        [
            call("c3"),
            call("c4"),
        ]
    )

    # LearningSolver calls after_solve
    component.after_solve(solver, instance, None, None)

    # Should query slack for all inequalities
    internal.get_inequality_slacks.assert_called_once()

    # Should store constraint slacks in instance object
    assert hasattr(instance, "slacks")
    assert instance.slacks == {
        "c1": 0.5,
        "c2": 0.0,
        "c3": 0.0,
        "c4": 1.4,
    }


def test_drop_redundant_with_check_dropped():
    solver, internal, instance, classifiers = _setup()

    component = DropRedundantInequalitiesStep(
        check_dropped=True, violation_tolerance=1e-3
    )
    component.classifiers = classifiers

    # LearningSolver call before_solve
    component.before_solve(solver, instance, None)

    # Assert constraints are extracted
    assert internal.extract_constraint.call_count == 2
    internal.extract_constraint.assert_has_calls(
        [
            call("c3"),
            call("c4"),
        ]
    )

    # LearningSolver calls iteration_cb (first time)
    should_repeat = component.iteration_cb(solver, instance, None)

    # Should ask LearningSolver to repeat
    assert should_repeat

    # Should ask solver if removed constraints are satisfied (mock always returns false)
    internal.is_constraint_satisfied.assert_has_calls(
        [
            call("<c3>", 1e-3),
            call("<c4>", 1e-3),
        ]
    )

    # Should add constraints back to LP relaxation
    internal.add_constraint.assert_has_calls([call("<c3>"), call("<c4>")])

    # LearningSolver calls iteration_cb (second time)
    should_repeat = component.iteration_cb(solver, instance, None)
    assert not should_repeat


def test_x_y_fit_predict_evaluate():
    instances = [Mock(spec=Instance), Mock(spec=Instance)]
    component = DropRedundantInequalitiesStep(slack_tolerance=0.05, threshold=0.80)
    component.classifiers = {
        "type-a": Mock(spec=Classifier),
        "type-b": Mock(spec=Classifier),
    }
    component.classifiers["type-a"].predict_proba = Mock(
        return_value=[
            [0.20, 0.80],
        ]
    )
    component.classifiers["type-b"].predict_proba = Mock(
        return_value=[
            [0.50, 0.50],
            [0.05, 0.95],
        ]
    )

    # First mock instance
    instances[0].slacks = {
        "c1": 0.00,
        "c2": 0.05,
        "c3": 0.00,
        "c4": 30.0,
    }
    instances[0].get_constraint_category = Mock(
        side_effect=lambda cid: {
            "c1": None,
            "c2": "type-a",
            "c3": "type-a",
            "c4": "type-b",
        }[cid]
    )
    instances[0].get_constraint_features = Mock(
        side_effect=lambda cid: {
            "c2": [1.0, 0.0],
            "c3": [0.5, 0.5],
            "c4": [1.0],
        }[cid]
    )

    # Second mock instance
    instances[1].slacks = {
        "c1": 0.00,
        "c3": 0.30,
        "c4": 0.00,
        "c5": 0.00,
    }
    instances[1].get_constraint_category = Mock(
        side_effect=lambda cid: {
            "c1": None,
            "c3": "type-a",
            "c4": "type-b",
            "c5": "type-b",
        }[cid]
    )
    instances[1].get_constraint_features = Mock(
        side_effect=lambda cid: {
            "c3": [0.3, 0.4],
            "c4": [0.7],
            "c5": [0.8],
        }[cid]
    )

    expected_x = {
        "type-a": [[1.0, 0.0], [0.5, 0.5], [0.3, 0.4]],
        "type-b": [[1.0], [0.7], [0.8]],
    }
    expected_y = {"type-a": [[0], [0], [1]], "type-b": [[1], [0], [0]]}

    # Should build X and Y matrices correctly
    assert component.x(instances) == expected_x
    assert component.y(instances) == expected_y

    # Should pass along X and Y matrices to classifiers
    component.fit(instances)
    component.classifiers["type-a"].fit.assert_called_with(
        expected_x["type-a"],
        expected_y["type-a"],
    )
    component.classifiers["type-b"].fit.assert_called_with(
        expected_x["type-b"],
        expected_y["type-b"],
    )

    assert component.predict(expected_x) == {"type-a": [[1]], "type-b": [[0], [1]]}

    ev = component.evaluate(instances[1])
    assert ev["True positive"] == 1
    assert ev["True negative"] == 1
    assert ev["False positive"] == 1
    assert ev["False negative"] == 0
