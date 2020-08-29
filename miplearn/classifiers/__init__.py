#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import ABC, abstractmethod

import numpy as np


class Classifier(ABC):
    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict_proba(self, x_test):
        pass

    def predict(self, x_test):
        proba = self.predict_proba(x_test)
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (x_test.shape[0], 2)
        return (proba[:, 1] > 0.5).astype(float)


class Regressor(ABC):
    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self):
        pass
