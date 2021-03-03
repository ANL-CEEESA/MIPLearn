#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.neighbors import KNeighborsClassifier

from miplearn import ScikitLearnClassifier


def test_constant_prediction():
    x_train = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    y_train = np.array(
        [
            [True, False],
            [True, False],
        ]
    )
    clf = ScikitLearnClassifier(
        KNeighborsClassifier(
            n_neighbors=1,
        )
    )
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_train)
    assert_array_equal(
        proba,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
    )
