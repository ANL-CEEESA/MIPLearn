#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Optional, Callable, List

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from miplearn.classifiers import Classifier, ScikitLearnClassifier

logger = logging.getLogger(__name__)


class CrossValidatedClassifier(Classifier):
    """
    A meta-classifier that, upon training, evaluates the performance of another
    candidate classifier on the training data set, using k-fold cross validation,
    then either adopts it, if its cv-score is high enough, or returns constant
    predictions for every x_test, otherwise.

    Parameters
    ----------
    classifier: Callable[[], ScikitLearnClassifier]
        A callable that constructs the candidate classifier.
    threshold: float
        Number from zero to one indicating how well must the candidate classifier
        perform to be adopted. The threshold is specified in comparison to a dummy
        classifier trained on the same dataset. For example, a threshold of 0.0
        indicates that any classifier as good as the dummy predictor is acceptable. A
        threshold of 1.0 indicates that only classifiers with perfect
        cross-validation scores are acceptable. Other numbers are a linear
        interpolation of these two extremes.
    constant: Optional[List[bool]]
        If the candidate classifier fails to meet the threshold, use a dummy classifier
        which always returns this prediction instead. The list should have exactly as
        many elements as the number of columns of `x_train` provided to `fit`.
    cv: int
        Number of folds.
    scoring: str
        Scoring function.
    """

    def __init__(
        self,
        classifier: Callable[[], ScikitLearnClassifier] = (
            lambda: ScikitLearnClassifier(LogisticRegression())
        ),
        threshold: float = 0.75,
        constant: Optional[List[bool]] = None,
        cv: int = 5,
        scoring: str = "accuracy",
    ):
        """"""
        super().__init__()
        if constant is None:
            constant = [True, False]
        self.n_classes = len(constant)
        self.classifier: Optional[ScikitLearnClassifier] = None
        self.classifier_factory = classifier
        self.constant: List[bool] = constant
        self.threshold = threshold
        self.cv = cv
        self.scoring = scoring

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        super().fit(x_train, y_train)
        (n_samples, n_classes) = x_train.shape
        assert n_classes == self.n_classes

        # Calculate dummy score and absolute score threshold
        y_train_avg = np.average(y_train)
        dummy_score = max(y_train_avg, 1 - y_train_avg)
        absolute_threshold = 1.0 * self.threshold + dummy_score * (1 - self.threshold)

        # Calculate cross validation score and decide which classifier to use
        clf = self.classifier_factory()
        assert clf is not None
        assert isinstance(clf, ScikitLearnClassifier), (
            f"The provided classifier callable must return a ScikitLearnClassifier. "
            f"Found {clf.__class__.__name__} instead. If this is a scikit-learn "
            f"classifier, you must wrap it with ScikitLearnClassifier."
        )

        cv_score = float(
            np.mean(
                cross_val_score(
                    clf.inner_clf,
                    x_train,
                    y_train[:, 1],
                    cv=self.cv,
                    scoring=self.scoring,
                )
            )
        )
        if cv_score >= absolute_threshold:
            logger.debug(
                "cv_score is above threshold (%.2f >= %.2f); keeping"
                % (cv_score, absolute_threshold)
            )
            self.classifier = clf
        else:
            logger.debug(
                "cv_score is below threshold (%.2f < %.2f); discarding"
                % (cv_score, absolute_threshold)
            )
            self.classifier = ScikitLearnClassifier(
                DummyClassifier(
                    strategy="constant",
                    constant=self.constant[1],
                )
            )

        # Train chosen classifier
        assert self.classifier is not None
        assert isinstance(self.classifier, ScikitLearnClassifier)
        self.classifier.fit(x_train, y_train)

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        super().predict_proba(x_test)
        assert self.classifier is not None
        return self.classifier.predict_proba(x_test)
