#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from copy import deepcopy
from typing import Dict, Callable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from miplearn.classifiers import Classifier, ScikitLearnClassifier
from miplearn.classifiers.counting import CountingClassifier

logger = logging.getLogger(__name__)


class CandidateClassifierSpecs:
    """
    Specifications describing how to construct a certain classifier, and under
    which circumstances it can be used.

    Parameters
    ----------
    min_samples: int
        Minimum number of samples for this classifier to be considered.
    classifier: Callable[[], Classifier]
        Callable that constructs the classifier.
    """

    def __init__(
        self,
        classifier: Callable[[], Classifier],
        min_samples: int = 0,
    ) -> None:
        self.min_samples = min_samples
        self.classifier = classifier


class AdaptiveClassifier(Classifier):
    """
    A meta-classifier which dynamically selects what actual classifier to use
    based on its cross-validation score on a particular training data set.

    Parameters
    ----------
    candidates: Dict[str, CandidateClassifierSpecs]
        A dictionary of candidate classifiers to consider, mapping the name of the
        candidate to its specs, which describes how to construct it and under what
        scenarios. If no candidates are provided, uses a fixed set of defaults,
        which includes `CountingClassifier`, `KNeighborsClassifier` and
        `LogisticRegression`.
    """

    def __init__(
        self,
        candidates: Dict[str, CandidateClassifierSpecs] = None,
    ) -> None:
        super().__init__()
        if candidates is None:
            candidates = {
                "knn(100)": CandidateClassifierSpecs(
                    classifier=lambda: ScikitLearnClassifier(
                        KNeighborsClassifier(n_neighbors=100)
                    ),
                    min_samples=100,
                ),
                "logistic": CandidateClassifierSpecs(
                    classifier=lambda: ScikitLearnClassifier(
                        make_pipeline(
                            StandardScaler(),
                            LogisticRegression(),
                        )
                    ),
                    min_samples=30,
                ),
                "counting": CandidateClassifierSpecs(
                    classifier=lambda: CountingClassifier(),
                ),
            }
        self.candidates = candidates
        self.classifier: Optional[Classifier] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        super().fit(x_train, y_train)
        n_samples = x_train.shape[0]
        assert y_train.shape == (n_samples, 2)

        best_name, best_clf, best_score = None, None, -float("inf")
        for (name, specs) in self.candidates.items():
            if n_samples < specs.min_samples:
                continue
            clf = specs.classifier()
            clf.fit(x_train, y_train)
            proba = clf.predict_proba(x_train)
            # FIXME: Switch to k-fold cross validation
            score = roc_auc_score(y_train[:, 1], proba[:, 1])
            if score > best_score:
                best_name, best_clf, best_score = name, clf, score
        logger.debug("Best classifier: %s (score=%.3f)" % (best_name, best_score))
        self.classifier = best_clf

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        super().predict_proba(x_test)
        assert self.classifier is not None
        return self.classifier.predict_proba(x_test)
