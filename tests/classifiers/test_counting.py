#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from numpy.linalg import norm

from miplearn.classifiers.counting import CountingClassifier

E = 0.1


def test_counting():
    clf = CountingClassifier()
    clf.fit(np.zeros((8, 25)), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    expected_proba = np.array([[0.375, 0.625], [0.375, 0.625]])
    actual_proba = clf.predict_proba(np.zeros((2, 25)))
    assert norm(actual_proba - expected_proba) < E
