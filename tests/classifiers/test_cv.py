#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from miplearn.classifiers import ScikitLearnClassifier
from miplearn.classifiers.cv import CrossValidatedClassifier

E = 0.1


def test_cv() -> None:
    # Training set: label is true if point is inside a 2D circle
    x_train = np.array(
        [
            [
                x1,
                x2,
            ]
            for x1 in range(-10, 11)
            for x2 in range(-10, 11)
        ]
    )
    x_train = StandardScaler().fit_transform(x_train)
    n_samples = x_train.shape[0]
    y_train = np.array(
        [
            [
                False,
                True,
            ]
            if x1 * x1 + x2 * x2 <= 100
            else [
                True,
                False,
            ]
            for x1 in range(-10, 11)
            for x2 in range(-10, 11)
        ]
    )

    # Support vector machines with linear kernels do not perform well on this
    # data set, so predictor should return the given constant.
    clf = CrossValidatedClassifier(
        classifier=lambda: ScikitLearnClassifier(
            SVC(
                probability=True,
                random_state=42,
            )
        ),
        threshold=0.90,
        constant=[True, False],
        cv=30,
    )
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_train)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (n_samples, 2)

    y_pred = (proba[:, 1] > 0.5).astype(float)
    assert norm(np.zeros(n_samples) - y_pred) < E

    # Support vector machines with quadratic kernels perform almost perfectly
    # on this data set, so predictor should return their prediction.
    clf = CrossValidatedClassifier(
        classifier=lambda: ScikitLearnClassifier(
            SVC(
                probability=True,
                kernel="poly",
                degree=2,
                random_state=42,
            )
        ),
        threshold=0.90,
        cv=30,
    )
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_train)
    y_pred = (proba[:, 1] > 0.5).astype(float)
    assert norm(y_train[:, 1] - y_pred) < E
