#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from unittest.mock import Mock

import numpy as np
from miplearn import DynamicLazyConstraintsComponent, LearningSolver, InternalSolver
from miplearn.classifiers import Classifier
from miplearn.tests import get_test_pyomo_instances
from numpy.linalg import norm

E = 0.1


def test_lazy_fit():
    instances, models = get_test_pyomo_instances()
    instances[0].found_violated_lazy_constraints = ["a", "b"]
    instances[1].found_violated_lazy_constraints = ["b", "c"]
    classifier = Mock(spec=Classifier)
    component = DynamicLazyConstraintsComponent(classifier=classifier)

    component.fit(instances)

    # Should create one classifier for each violation
    assert "a" in component.classifiers
    assert "b" in component.classifiers
    assert "c" in component.classifiers

    # Should provide correct x_train to each classifier
    expected_x_train_a = np.array([[67., 21.75, 1287.92], [70., 23.75, 1199.83]])
    expected_x_train_b = np.array([[67., 21.75, 1287.92], [70., 23.75, 1199.83]])
    expected_x_train_c = np.array([[67., 21.75, 1287.92], [70., 23.75, 1199.83]])
    actual_x_train_a = component.classifiers["a"].fit.call_args[0][0]
    actual_x_train_b = component.classifiers["b"].fit.call_args[0][0]
    actual_x_train_c = component.classifiers["c"].fit.call_args[0][0]
    assert norm(expected_x_train_a - actual_x_train_a) < E
    assert norm(expected_x_train_b - actual_x_train_b) < E
    assert norm(expected_x_train_c - actual_x_train_c) < E

    # Should provide correct y_train to each classifier
    expected_y_train_a = np.array([1.0, 0.0])
    expected_y_train_b = np.array([1.0, 1.0])
    expected_y_train_c = np.array([0.0, 1.0])
    actual_y_train_a = component.classifiers["a"].fit.call_args[0][1]
    actual_y_train_b = component.classifiers["b"].fit.call_args[0][1]
    actual_y_train_c = component.classifiers["c"].fit.call_args[0][1]
    assert norm(expected_y_train_a - actual_y_train_a) < E
    assert norm(expected_y_train_b - actual_y_train_b) < E
    assert norm(expected_y_train_c - actual_y_train_c) < E


def test_lazy_before():
    instances, models = get_test_pyomo_instances()
    instances[0].build_lazy_constraint = Mock(return_value="c1")
    solver = LearningSolver()
    solver.internal_solver = Mock(spec=InternalSolver)
    component = DynamicLazyConstraintsComponent(threshold=0.10)
    component.classifiers = {"a": Mock(spec=Classifier),
                             "b": Mock(spec=Classifier)}
    component.classifiers["a"].predict_proba = Mock(return_value=[[0.95, 0.05]])
    component.classifiers["b"].predict_proba = Mock(return_value=[[0.02, 0.80]])

    component.before_solve(solver, instances[0], models[0])

    # Should ask classifier likelihood of each constraint being violated
    expected_x_test_a = np.array([[67., 21.75, 1287.92]])
    expected_x_test_b = np.array([[67., 21.75, 1287.92]])
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
    component.classifiers = {"a": Mock(spec=Classifier),
                             "b": Mock(spec=Classifier),
                             "c": Mock(spec=Classifier)}
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
            "Recall": 2/3.,
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
        }
    }
    