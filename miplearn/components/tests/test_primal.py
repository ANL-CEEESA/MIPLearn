#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from unittest.mock import Mock

import numpy as np
from miplearn import PrimalSolutionComponent
from miplearn.classifiers import Classifier
from miplearn.tests import get_test_pyomo_instances


def test_predict():
    instances, models = get_test_pyomo_instances()
    comp = PrimalSolutionComponent()
    comp.fit(instances)
    solution = comp.predict(instances[0])
    assert "x" in solution
    assert 0 in solution["x"]
    assert 1 in solution["x"]
    assert 2 in solution["x"]
    assert 3 in solution["x"]


def test_evaluate():
    instances, models = get_test_pyomo_instances()
    clf_zero = Mock(spec=Classifier)
    clf_zero.predict_proba = Mock(return_value=np.array([
        [0., 1.],  # x[0]
        [0., 1.],  # x[1]
        [1., 0.],  # x[2]
        [1., 0.],  # x[3]
    ]))
    clf_one = Mock(spec=Classifier)
    clf_one.predict_proba = Mock(return_value=np.array([
        [1., 0.],  # x[0] instances[0]
        [1., 0.],  # x[1] instances[0]
        [0., 1.],  # x[2] instances[0]
        [1., 0.],  # x[3] instances[0]
    ]))
    comp = PrimalSolutionComponent(classifier=[clf_zero, clf_one],
                                   threshold=0.50)
    comp.fit(instances[:1])
    assert comp.predict(instances[0]) == {"x": {0: 0,
                                                1: 0,
                                                2: 1,
                                                3: None}}
    assert instances[0].solution == {"x": {0: 1,
                                           1: 0,
                                           2: 1,
                                           3: 1}}
    ev = comp.evaluate(instances[:1])
    assert ev == {'Fix one': {0: {'Accuracy': 0.5,
                                  'Condition negative': 1,
                                  'Condition negative (%)': 25.0,
                                  'Condition positive': 3,
                                  'Condition positive (%)': 75.0,
                                  'F1 score': 0.5,
                                  'False negative': 2,
                                  'False negative (%)': 50.0,
                                  'False positive': 0,
                                  'False positive (%)': 0.0,
                                  'Precision': 1.0,
                                  'Predicted negative': 3,
                                  'Predicted negative (%)': 75.0,
                                  'Predicted positive': 1,
                                  'Predicted positive (%)': 25.0,
                                  'Recall': 0.3333333333333333,
                                  'True negative': 1,
                                  'True negative (%)': 25.0,
                                  'True positive': 1,
                                  'True positive (%)': 25.0}},
                  'Fix zero': {0: {'Accuracy': 0.75,
                                   'Condition negative': 3,
                                   'Condition negative (%)': 75.0,
                                   'Condition positive': 1,
                                   'Condition positive (%)': 25.0,
                                   'F1 score': 0.6666666666666666,
                                   'False negative': 0,
                                   'False negative (%)': 0.0,
                                   'False positive': 1,
                                   'False positive (%)': 25.0,
                                   'Precision': 0.5,
                                   'Predicted negative': 2,
                                   'Predicted negative (%)': 50.0,
                                   'Predicted positive': 2,
                                   'Predicted positive (%)': 50.0,
                                   'Recall': 1.0,
                                   'True negative': 2,
                                   'True negative (%)': 50.0,
                                   'True positive': 1,
                                   'True positive (%)': 25.0}}}


def test_primal_parallel_fit():
    instances, models = get_test_pyomo_instances()
    comp = PrimalSolutionComponent()
    comp.fit(instances, n_jobs=2)
    assert len(comp.classifiers) == 2
