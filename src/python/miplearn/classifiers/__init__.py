#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict_proba(self, x_test):
        pass


class Regressor(ABC):
    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self):
        pass
