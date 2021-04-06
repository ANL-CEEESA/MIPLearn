#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from numpy.linalg import norm
from sklearn.svm import SVC

from miplearn.classifiers.adaptive import CandidateClassifierSpecs, AdaptiveClassifier
from miplearn.classifiers.sklearn import ScikitLearnClassifier
from tests.classifiers import _build_circle_training_data


def test_adaptive() -> None:
    clf = AdaptiveClassifier(
        candidates={
            "linear": CandidateClassifierSpecs(
                classifier=ScikitLearnClassifier(
                    SVC(
                        probability=True,
                        random_state=42,
                    )
                )
            ),
            "poly": CandidateClassifierSpecs(
                classifier=ScikitLearnClassifier(
                    SVC(
                        probability=True,
                        kernel="poly",
                        degree=2,
                        random_state=42,
                    )
                )
            ),
        }
    )
    x_train, y_train = _build_circle_training_data()
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_train)
    y_pred = (proba[:, 1] > 0.5).astype(float)
    assert norm(y_train[:, 1] - y_pred) < 0.1
