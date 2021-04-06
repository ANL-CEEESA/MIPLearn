#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from numpy.linalg import norm

from miplearn.classifiers.counting import CountingClassifier

E = 0.1


def test_counting():
    clf = CountingClassifier()
    n_features = 25
    x_train = np.zeros((8, n_features))
    y_train = np.array(
        [
            [True, False, False],
            [True, False, False],
            [False, True, False],
            [True, False, False],
            [False, True, False],
            [False, True, False],
            [False, True, False],
            [False, False, True],
        ]
    )
    x_test = np.zeros((2, n_features))
    y_expected = np.array(
        [
            [3 / 8.0, 4 / 8.0, 1 / 8.0],
            [3 / 8.0, 4 / 8.0, 1 / 8.0],
        ]
    )
    clf.fit(x_train, y_train)
    y_actual = clf.predict_proba(x_test)
    assert norm(y_actual - y_expected) < E
