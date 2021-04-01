#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from miplearn.classifiers.sklearn import ScikitLearnClassifier, ScikitLearnRegressor


def test_constant_prediction():
    x_train = np.array([[0.0, 1.0], [1.0, 0.0]])
    y_train = np.array([[True, False], [True, False]])
    clf = ScikitLearnClassifier(KNeighborsClassifier(n_neighbors=1))
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_train)
    assert_array_equal(
        proba,
        np.array([[1.0, 0.0], [1.0, 0.0]]),
    )


def test_regressor():
    x_train = np.array([[0.0, 1.0], [1.0, 4.0], [2.0, 2.0]])
    y_train = np.array([[1.0], [5.0], [4.0]])
    x_test = np.array([[4.0, 4.0], [0.0, 0.0]])
    clf = ScikitLearnRegressor(LinearRegression())
    clf.fit(x_train, y_train)
    y_test_actual = clf.predict(x_test)
    y_test_expected = np.array([[8.0], [0.0]])
    assert_array_equal(np.round(y_test_actual, 2), y_test_expected)
