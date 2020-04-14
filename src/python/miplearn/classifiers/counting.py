#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn.classifiers import Classifier
import numpy as np


class CountingClassifier(Classifier):
    """
    A classifier that generates constant predictions, based only on the
    frequency of the training labels. For example, if y_train is [1.0, 0.0, 0.0]
    this classifier always returns [0.66 0.33] for any x_test. It essentially
    counts how many times each label appeared, hence the name.
    """

    def __init__(self):
        self.mean = None

    def fit(self, x_train, y_train):
        self.mean = np.mean(y_train)

    def predict_proba(self, x_test):
        return np.array([[1 - self.mean, self.mean]])

    def __repr__(self):
        return "CountingClassifier(mean=%.3f)" % self.mean
