#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from copy import deepcopy

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from miplearn.classifiers import Classifier

logger = logging.getLogger(__name__)


class CrossValidatedClassifier(Classifier):
    """
    A meta-classifier that, upon training, evaluates the performance of another
    classifier on the training data set using k-fold cross validation, then
    either adopts the other classifier it if the cv-score is high enough, or
    returns a constant label for every x_test otherwise.

    The threshold is specified in comparison to a dummy classifier trained
    on the same dataset. For example, a threshold of 0.0 indicates that any
    classifier as good as the dummy predictor is acceptable. A threshold of 1.0
    indicates that only classifier with a perfect cross-validation score are
    acceptable. Other numbers are a linear interpolation of these two extremes.
    """

    def __init__(
        self,
        classifier=LogisticRegression(),
        threshold=0.75,
        constant=0.0,
        cv=5,
        scoring="accuracy",
    ):
        super().__init__()
        self.classifier = None
        self.classifier_prototype = classifier
        self.constant = constant
        self.threshold = threshold
        self.cv = cv
        self.scoring = scoring

    def fit(self, x_train, y_train):
        # super().fit(x_train, y_train)

        # Calculate dummy score and absolute score threshold
        y_train_avg = np.average(y_train)
        dummy_score = max(y_train_avg, 1 - y_train_avg)
        absolute_threshold = 1.0 * self.threshold + dummy_score * (1 - self.threshold)

        # Calculate cross validation score and decide which classifier to use
        clf = deepcopy(self.classifier_prototype)
        cv_score = float(
            np.mean(
                cross_val_score(
                    clf,
                    x_train,
                    y_train,
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
            self.classifier = DummyClassifier(
                strategy="constant",
                constant=self.constant,
            )

        # Train chosen classifier
        self.classifier.fit(x_train, y_train)

    def predict_proba(self, x_test):
        # super().predict_proba(x_test)
        return self.classifier.predict_proba(x_test)
