#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import List, cast
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.linalg import norm
from numpy.testing import assert_array_equal

from miplearn import Instance
from miplearn.classifiers import Classifier
from miplearn.components.lazy_dynamic import DynamicLazyConstraintsComponent
from miplearn.features import (
    TrainingSample,
    Features,
    ConstraintFeatures,
    InstanceFeatures,
)
from miplearn.solvers.internal import InternalSolver
from miplearn.solvers.learning import LearningSolver
from tests.fixtures.knapsack import get_test_pyomo_instances

E = 0.1


def test_lazy_fit():
    instances, models = get_test_pyomo_instances()
    instances[0].found_violated_lazy_constraints = ["a", "b"]
    instances[1].found_violated_lazy_constraints = ["b", "c"]
    classifier = Mock(spec=Classifier)
    classifier.clone = lambda: Mock(spec=Classifier)
    component = DynamicLazyConstraintsComponent(classifier=classifier)

    component.fit(instances)

    # Should create one classifier for each violation
    assert "a" in component.classifiers
    assert "b" in component.classifiers
    assert "c" in component.classifiers

    # Should provide correct x_train to each classifier
    expected_x_train_a = np.array([[67.0, 21.75, 1287.92], [70.0, 23.75, 1199.83]])
    expected_x_train_b = np.array([[67.0, 21.75, 1287.92], [70.0, 23.75, 1199.83]])
    expected_x_train_c = np.array([[67.0, 21.75, 1287.92], [70.0, 23.75, 1199.83]])
    actual_x_train_a = component.classifiers["a"].fit.call_args[0][0]
    actual_x_train_b = component.classifiers["b"].fit.call_args[0][0]
    actual_x_train_c = component.classifiers["c"].fit.call_args[0][0]
    assert norm(expected_x_train_a - actual_x_train_a) < E
    assert norm(expected_x_train_b - actual_x_train_b) < E
    assert norm(expected_x_train_c - actual_x_train_c) < E

    # Should provide correct y_train to each classifier
    expected_y_train_a = np.array(
        [
            [False, True],
            [True, False],
        ]
    )
    expected_y_train_b = np.array(
        [
            [False, True],
            [False, True],
        ]
    )
    expected_y_train_c = np.array(
        [
            [True, False],
            [False, True],
        ]
    )
    assert_array_equal(
        component.classifiers["a"].fit.call_args[0][1],
        expected_y_train_a,
    )
    assert_array_equal(
        component.classifiers["b"].fit.call_args[0][1],
        expected_y_train_b,
    )
    assert_array_equal(
        component.classifiers["c"].fit.call_args[0][1],
        expected_y_train_c,
    )


def test_lazy_before():
    instances, models = get_test_pyomo_instances()
    instances[0].build_lazy_constraint = Mock(return_value="c1")
    solver = LearningSolver()
    solver.internal_solver = Mock(spec=InternalSolver)
    component = DynamicLazyConstraintsComponent(threshold=0.10)
    component.classifiers = {"a": Mock(spec=Classifier), "b": Mock(spec=Classifier)}
    component.classifiers["a"].predict_proba = Mock(return_value=[[0.95, 0.05]])
    component.classifiers["b"].predict_proba = Mock(return_value=[[0.02, 0.80]])

    component.before_solve_mip(
        solver=solver,
        instance=instances[0],
        model=models[0],
        stats=None,
        features=None,
        training_data=None,
    )

    # Should ask classifier likelihood of each constraint being violated
    expected_x_test_a = np.array([[67.0, 21.75, 1287.92]])
    expected_x_test_b = np.array([[67.0, 21.75, 1287.92]])
    actual_x_test_a = component.classifiers["a"].predict_proba.call_args[0][0]
    actual_x_test_b = component.classifiers["b"].predict_proba.call_args[0][0]
    assert norm(expected_x_test_a - actual_x_test_a) < E
    assert norm(expected_x_test_b - actual_x_test_b) < E

    # Should ask instance to generate cut for constraints whose likelihood
    # of being violated exceeds the threshold
    instances[0].build_lazy_constraint.assert_called_once_with(models[0], "b")

    # Should ask internal solver to add generated constraint
    solver.internal_solver.add_constraint.assert_called_once_with("c1")


def test_lazy_evaluate():
    instances, models = get_test_pyomo_instances()
    component = DynamicLazyConstraintsComponent()
    component.classifiers = {
        "a": Mock(spec=Classifier),
        "b": Mock(spec=Classifier),
        "c": Mock(spec=Classifier),
    }
    component.classifiers["a"].predict_proba = Mock(return_value=[[1.0, 0.0]])
    component.classifiers["b"].predict_proba = Mock(return_value=[[0.0, 1.0]])
    component.classifiers["c"].predict_proba = Mock(return_value=[[0.0, 1.0]])

    instances[0].found_violated_lazy_constraints = ["a", "b", "c"]
    instances[1].found_violated_lazy_constraints = ["b", "d"]
    assert component.evaluate(instances) == {
        0: {
            "Accuracy": 0.75,
            "F1 score": 0.8,
            "Precision": 1.0,
            "Recall": 2 / 3.0,
            "Predicted positive": 2,
            "Predicted negative": 2,
            "Condition positive": 3,
            "Condition negative": 1,
            "False negative": 1,
            "False positive": 0,
            "True negative": 1,
            "True positive": 2,
            "Predicted positive (%)": 50.0,
            "Predicted negative (%)": 50.0,
            "Condition positive (%)": 75.0,
            "Condition negative (%)": 25.0,
            "False negative (%)": 25.0,
            "False positive (%)": 0,
            "True negative (%)": 25.0,
            "True positive (%)": 50.0,
        },
        1: {
            "Accuracy": 0.5,
            "F1 score": 0.5,
            "Precision": 0.5,
            "Recall": 0.5,
            "Predicted positive": 2,
            "Predicted negative": 2,
            "Condition positive": 2,
            "Condition negative": 2,
            "False negative": 1,
            "False positive": 1,
            "True negative": 1,
            "True positive": 1,
            "Predicted positive (%)": 50.0,
            "Predicted negative (%)": 50.0,
            "Condition positive (%)": 50.0,
            "Condition negative (%)": 50.0,
            "False negative (%)": 25.0,
            "False positive (%)": 25.0,
            "True negative (%)": 25.0,
            "True positive (%)": 25.0,
        },
    }


@pytest.fixture
def training_instances() -> List[Instance]:
    instances = [cast(Instance, Mock(spec=Instance)) for _ in range(2)]
    instances[0].features = Features(
        instance=InstanceFeatures(
            user_features=[50.0],
        ),
    )
    instances[0].training_data = [
        TrainingSample(lazy_enforced={"c1", "c2"}),
        TrainingSample(lazy_enforced={"c2", "c3"}),
    ]
    instances[0].get_constraint_category = Mock(  # type: ignore
        side_effect=lambda cid: {
            "c1": "type-a",
            "c2": "type-a",
            "c3": "type-b",
            "c4": "type-b",
        }[cid]
    )
    instances[0].get_constraint_features = Mock(  # type: ignore
        side_effect=lambda cid: {
            "c1": [1.0, 2.0, 3.0],
            "c2": [4.0, 5.0, 6.0],
            "c3": [1.0, 2.0],
            "c4": [3.0, 4.0],
        }[cid]
    )
    instances[1].features = Features(
        instance=InstanceFeatures(
            user_features=[80.0],
        ),
    )
    instances[1].training_data = [
        TrainingSample(lazy_enforced={"c3", "c4"}),
    ]
    instances[1].get_constraint_category = Mock(  # type: ignore
        side_effect=lambda cid: {
            "c1": None,
            "c2": "type-a",
            "c3": "type-b",
            "c4": "type-b",
        }[cid]
    )
    instances[1].get_constraint_features = Mock(  # type: ignore
        side_effect=lambda cid: {
            "c2": [7.0, 8.0, 9.0],
            "c3": [5.0, 6.0],
            "c4": [7.0, 8.0],
        }[cid]
    )
    return instances


def test_fit_new(training_instances: List[Instance]) -> None:
    clf = Mock(spec=Classifier)
    clf.clone = Mock(side_effect=lambda: Mock(spec=Classifier))
    comp = DynamicLazyConstraintsComponent(classifier=clf)
    comp.fit_new(training_instances)
    assert clf.clone.call_count == 2

    assert "type-a" in comp.classifiers
    clf_a = comp.classifiers["type-a"]
    assert clf_a.fit.call_count == 1  # type: ignore
    assert_array_equal(
        clf_a.fit.call_args[0][0],  # type: ignore
        np.array(
            [
                [50.0, 1.0, 2.0, 3.0],
                [50.0, 4.0, 5.0, 6.0],
                [50.0, 1.0, 2.0, 3.0],
                [50.0, 4.0, 5.0, 6.0],
                [80.0, 7.0, 8.0, 9.0],
            ]
        ),
    )
    assert_array_equal(
        clf_a.fit.call_args[0][1],  # type: ignore
        np.array(
            [
                [False, True],
                [False, True],
                [True, False],
                [False, True],
                [True, False],
            ]
        ),
    )

    assert "type-b" in comp.classifiers
    clf_b = comp.classifiers["type-b"]
    assert clf_b.fit.call_count == 1  # type: ignore
    assert_array_equal(
        clf_b.fit.call_args[0][0],  # type: ignore
        np.array(
            [
                [50.0, 1.0, 2.0],
                [50.0, 3.0, 4.0],
                [50.0, 1.0, 2.0],
                [50.0, 3.0, 4.0],
                [80.0, 5.0, 6.0],
                [80.0, 7.0, 8.0],
            ]
        ),
    )
    assert_array_equal(
        clf_b.fit.call_args[0][1],  # type: ignore
        np.array(
            [
                [True, False],
                [True, False],
                [False, True],
                [True, False],
                [False, True],
                [False, True],
            ]
        ),
    )
