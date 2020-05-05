#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from miplearn.classifiers.evaluator import ClassifierEvaluator
from sklearn.neighbors import KNeighborsClassifier


def test_evaluator():
    clf_a = KNeighborsClassifier(n_neighbors=1)
    clf_b = KNeighborsClassifier(n_neighbors=2)
    x_train = np.array([[0, 0], [1, 0]])
    y_train = np.array([0, 1])
    clf_a.fit(x_train, y_train)
    clf_b.fit(x_train, y_train)
    ev = ClassifierEvaluator()
    assert ev.evaluate(clf_a, x_train, y_train) == 1.0
    assert ev.evaluate(clf_b, x_train, y_train) == 0.5

