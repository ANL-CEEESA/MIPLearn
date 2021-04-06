#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from unittest.mock import Mock, call

import numpy as np

from miplearn.classifiers import Classifier
from miplearn.components.steps.drop_redundant import DropRedundantInequalitiesStep
from miplearn.components.steps.relax_integrality import RelaxIntegralityStep
from miplearn.instance import Instance
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.internal import InternalSolver
from miplearn.solvers.learning import LearningSolver
from miplearn.features import TrainingSample, Features
from tests.fixtures.infeasible import get_infeasible_instance
from tests.fixtures.redundant import get_instance_with_redundancy


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
    internal.is_infeasible = Mock(return_value=False)

    instance = Mock(spec=Instance)
    instance.get_constraint_features = Mock(
        side_effect=lambda cid: {
            "c2": np.array([1.0, 0.0]),
            "c3": np.array([0.5, 0.5]),
            "c4": np.array([1.0]),
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
        return_value=np.array(
            [
                [0.20, 0.80],
                [0.05, 0.95],
            ]
        )
    )
    classifiers["type-b"].predict_proba = Mock(
        return_value=np.array(
            [
                [0.02, 0.98],
            ]
        )
    )

    return solver, internal, instance, classifiers


def test_drop_redundant():
    solver, internal, instance, classifiers = _setup()

    component = DropRedundantInequalitiesStep()
    component.classifiers = classifiers

    # LearningSolver calls before_solve
    component.before_solve_mip(
        solver=solver,
        instance=instance,
        model=None,
        stats={},
        features=Features(),
        training_data=TrainingSample(),
    )

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
    type_a_actual = component.classifiers["type-a"].predict_proba.call_args[0][0]
    type_b_actual = component.classifiers["type-b"].predict_proba.call_args[0][0]
    np.testing.assert_array_equal(type_a_actual, np.array([[1.0, 0.0], [0.5, 0.5]]))
    np.testing.assert_array_equal(type_b_actual, np.array([[1.0]]))

    # Should ask internal solver to remove constraints predicted as redundant
    assert internal.extract_constraint.call_count == 2
    internal.extract_constraint.assert_has_calls(
        [
            call("c3"),
            call("c4"),
        ]
    )

    # LearningSolver calls after_solve
    training_data = TrainingSample()
    component.after_solve_mip(
        solver=solver,
        instance=instance,
        model=None,
        stats={},
        features=Features(),
        training_data=training_data,
    )

    # Should query slack for all inequalities
    internal.get_inequality_slacks.assert_called_once()

    # Should store constraint slacks in instance object
    assert training_data.slacks == {
        "c1": 0.5,
        "c2": 0.0,
        "c3": 0.0,
        "c4": 1.4,
    }


def test_drop_redundant_with_check_feasibility():
    solver, internal, instance, classifiers = _setup()

    component = DropRedundantInequalitiesStep(
        check_feasibility=True,
        violation_tolerance=1e-3,
    )
    component.classifiers = classifiers

    # LearningSolver call before_solve
    component.before_solve_mip(
        solver=solver,
        instance=instance,
        model=None,
        stats={},
        features=Features(),
        training_data=TrainingSample(),
    )

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
            np.array([0.20, 0.80]),
        ]
    )
    component.classifiers["type-b"].predict_proba = Mock(
        return_value=np.array(
            [
                [0.50, 0.50],
                [0.05, 0.95],
            ]
        )
    )

    # First mock instance
    instances[0].training_data = [
        TrainingSample(
            slacks={
                "c1": 0.00,
                "c2": 0.05,
                "c3": 0.00,
                "c4": 30.0,
            }
        )
    ]
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
            "c2": np.array([1.0, 0.0]),
            "c3": np.array([0.5, 0.5]),
            "c4": np.array([1.0]),
        }[cid]
    )

    # Second mock instance
    instances[1].training_data = [
        TrainingSample(
            slacks={
                "c1": 0.00,
                "c3": 0.30,
                "c4": 0.00,
                "c5": 0.00,
            }
        )
    ]
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
            "c3": np.array([0.3, 0.4]),
            "c4": np.array([0.7]),
            "c5": np.array([0.8]),
        }[cid]
    )

    expected_x = {
        "type-a": np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.3, 0.4],
            ]
        ),
        "type-b": np.array(
            [
                [1.0],
                [0.7],
                [0.8],
            ]
        ),
    }
    expected_y = {
        "type-a": np.array(
            [
                [True, False],
                [True, False],
                [False, True],
            ]
        ),
        "type-b": np.array(
            [
                [False, True],
                [True, False],
                [True, False],
            ]
        ),
    }

    # Should build X and Y matrices correctly
    actual_x, actual_y = component.x_y(instances)
    for category in ["type-a", "type-b"]:
        np.testing.assert_array_equal(actual_x[category], expected_x[category])
        np.testing.assert_array_equal(actual_y[category], expected_y[category])

    # Should pass along X and Y matrices to classifiers
    component.fit(instances)
    for category in ["type-a", "type-b"]:
        actual_x = component.classifiers[category].fit.call_args[0][0]
        actual_y = component.classifiers[category].fit.call_args[0][1]
        np.testing.assert_array_equal(actual_x, expected_x[category])
        np.testing.assert_array_equal(actual_y, expected_y[category])

    assert component.predict(expected_x) == {
        "type-a": [
            [False, True],
        ],
        "type-b": [
            [True, False],
            [False, True],
        ],
    }

    ev = component.evaluate(instances[1])
    assert ev["True positive"] == 1
    assert ev["True negative"] == 1
    assert ev["False positive"] == 1
    assert ev["False negative"] == 0


def test_x_multiple_solves():
    instance = Mock(spec=Instance)
    instance.training_data = [
        TrainingSample(
            slacks={
                "c1": 0.00,
                "c2": 0.05,
                "c3": 0.00,
                "c4": 30.0,
            }
        ),
        TrainingSample(
            slacks={
                "c1": 0.00,
                "c2": 0.00,
                "c3": 1.00,
                "c4": 0.0,
            }
        ),
    ]
    instance.get_constraint_category = Mock(
        side_effect=lambda cid: {
            "c1": None,
            "c2": "type-a",
            "c3": "type-a",
            "c4": "type-b",
        }[cid]
    )
    instance.get_constraint_features = Mock(
        side_effect=lambda cid: {
            "c2": np.array([1.0, 0.0]),
            "c3": np.array([0.5, 0.5]),
            "c4": np.array([1.0]),
        }[cid]
    )

    expected_x = {
        "type-a": np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [1.0, 0.0],
                [0.5, 0.5],
            ]
        ),
        "type-b": np.array(
            [
                [1.0],
                [1.0],
            ]
        ),
    }

    expected_y = {
        "type-a": np.array(
            [
                [False, True],
                [True, False],
                [True, False],
                [False, True],
            ]
        ),
        "type-b": np.array(
            [
                [False, True],
                [True, False],
            ]
        ),
    }

    # Should build X and Y matrices correctly
    component = DropRedundantInequalitiesStep()
    actual_x, actual_y = component.x_y([instance])
    for category in ["type-a", "type-b"]:
        np.testing.assert_array_equal(actual_x[category], expected_x[category])
        np.testing.assert_array_equal(actual_y[category], expected_y[category])


def test_usage():
    for internal_solver in [GurobiSolver]:
        for instance in [
            get_instance_with_redundancy(internal_solver),
            get_infeasible_instance(internal_solver),
        ]:
            solver = LearningSolver(
                solver=internal_solver,
                components=[
                    RelaxIntegralityStep(),
                    DropRedundantInequalitiesStep(),
                ],
            )
            # The following should not crash
            solver.solve(instance)
            solver.fit([instance])
            solver.solve(instance)
