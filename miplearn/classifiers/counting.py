#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np

from miplearn.classifiers import Classifier


class CountingClassifier(Classifier):
    """
    A classifier that generates constant predictions, based only on the
    frequency of the training labels. For example, if y_train is [1.0, 0.0, 0.0]
    this classifier always returns [0.66 0.33] for any x_test. It essentially
    counts how many times each label appeared, hence the name.
    """

    def __init__(self) -> None:
        self.mean = None

    def fit(self, x_train, y_train):
        self.mean = np.mean(y_train)

    def predict_proba(self, x_test):
        return np.array([[1 - self.mean, self.mean] for _ in range(x_test.shape[0])])

    def __repr__(self):
        return "CountingClassifier(mean=%s)" % self.mean
