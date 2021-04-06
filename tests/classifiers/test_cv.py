#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from numpy.linalg import norm
from sklearn.svm import SVC

from miplearn.classifiers.cv import CrossValidatedClassifier
from miplearn.classifiers.sklearn import ScikitLearnClassifier
from tests.classifiers import _build_circle_training_data

E = 0.1


def test_cv() -> None:
    x_train, y_train = _build_circle_training_data()
    n_samples = x_train.shape[0]

    # Support vector machines with linear kernels do not perform well on this
    # data set, so predictor should return the given constant.
    clf = CrossValidatedClassifier(
        classifier=ScikitLearnClassifier(
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
        classifier=ScikitLearnClassifier(
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
