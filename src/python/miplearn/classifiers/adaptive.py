#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from copy import deepcopy

from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AdaptiveClassifier(Classifier):
    """
    A meta-classifier which dynamically selects what actual classifier to use
    based on the number of samples in the training data.

    By default, uses CountingClassifier for less than 30 samples and
    LogisticRegression (with standard scaling) for 30 or more samples.
    """

    def __init__(self, classifiers=None):
        """
        Initializes the classifier.

        The `classifiers` argument must be a list of tuples where the second element
        of the tuple is the classifier and the first element is the number of
        samples required. For example, if `classifiers` is set to
        ```
            [(100, ClassifierA()),
             (50,  ClassifierB()),
             (0,   ClassifierC())]
        ``` then ClassifierA will be used if n_samples >= 100, ClassifierB will
        be used if 100 > n_samples >= 50 and ClassifierC will be used if
        50 > n_samples. The list must be ordered in (strictly) decreasing order.
        """
        if classifiers is None:
            classifiers = [
                (30, make_pipeline(StandardScaler(), LogisticRegression())),
                (0, CountingClassifier())
            ]
        self.available_classifiers = classifiers
        self.classifier = None

    def fit(self, x_train, y_train):
        n_samples = x_train.shape[0]

        for (min_samples, clf_prototype) in self.available_classifiers:
            if n_samples >= min_samples:
                self.classifier = deepcopy(clf_prototype)
                self.classifier.fit(x_train, y_train)
                break

    def predict_proba(self, x_test):
        return self.classifier.predict_proba(x_test)


