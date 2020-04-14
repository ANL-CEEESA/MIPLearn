#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from copy import deepcopy

import numpy as np

from miplearn.classifiers import Classifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class AdaptiveClassifier(Classifier):
    """
    A classifier that automatically switches strategies based on the number of
    samples and cross-validation scores.
    """
    def __init__(self,
                 predictor=None,
                 min_samples_predict=1,
                 min_samples_cv=100,
                 thr_fix=0.999,
                 thr_alpha=0.50,
                 thr_balance=0.95,
                 ):
        self.min_samples_predict = min_samples_predict
        self.min_samples_cv = min_samples_cv
        self.thr_fix = thr_fix
        self.thr_alpha = thr_alpha
        self.thr_balance = thr_balance
        self.predictor_factory = predictor
        self.predictor = None

    def fit(self, x_train, y_train):
        n_samples = x_train.shape[0]

        # If number of samples is too small, don't predict anything.
        if n_samples < self.min_samples_predict:
            logger.debug("    Too few samples (%d); always predicting false" % n_samples)
            self.predictor = 0
            return

        # If vast majority of observations are false, always return false.
        y_train_avg = np.average(y_train)
        if y_train_avg <= 1.0 - self.thr_fix:
            logger.debug("    Most samples are negative (%.3f); always returning false" % y_train_avg)
            self.predictor = 0
            return

        # If vast majority of observations are true, always return true.
        if y_train_avg >= self.thr_fix:
            logger.debug("    Most samples are positive (%.3f); always returning true" % y_train_avg)
            self.predictor = 1
            return

        # If classes are too unbalanced, don't predict anything.
        if y_train_avg < (1 - self.thr_balance) or y_train_avg > self.thr_balance:
            logger.debug("    Classes are too unbalanced (%.3f); always returning false" % y_train_avg)
            self.predictor = 0
            return

        # Select ML model if none is provided
        if self.predictor_factory is None:
            if n_samples < 30:
                from sklearn.neighbors import KNeighborsClassifier
                self.predictor_factory = KNeighborsClassifier(n_neighbors=n_samples)
            else:
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                self.predictor_factory = make_pipeline(StandardScaler(), LogisticRegression())

        # Create predictor
        if callable(self.predictor_factory):
            pred = self.predictor_factory()
        else:
            pred = deepcopy(self.predictor_factory)

        # Skip cross-validation if number of samples is too small
        if n_samples < self.min_samples_cv:
            logger.debug("    Too few samples (%d); skipping cross validation" % n_samples)
            self.predictor = pred
            self.predictor.fit(x_train, y_train)
            return

        # Calculate cross-validation score
        cv_score = np.mean(cross_val_score(pred, x_train, y_train, cv=5))
        dummy_score = max(y_train_avg, 1 - y_train_avg)
        cv_thr = 1. * self.thr_alpha + dummy_score * (1 - self.thr_alpha)

        # If cross-validation score is too low, don't predict anything.
        if cv_score < cv_thr:
            logger.debug("    Score is too low (%.3f < %.3f); always returning false" % (cv_score, cv_thr))
            self.predictor = 0
        else:
            logger.debug("    Score is acceptable (%.3f > %.3f); training classifier" % (cv_score, cv_thr))
            self.predictor = pred
            self.predictor.fit(x_train, y_train)

    def predict_proba(self, x_test):
        if isinstance(self.predictor, int):
            y_pred = np.zeros((x_test.shape[0], 2))
            y_pred[:, self.predictor] = 1.0
            return y_pred
        else:
            return self.predictor.predict_proba(x_test)

