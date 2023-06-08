#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Callable, Optional

import numpy as np
import sklearn.base
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels


class SingleClassFix(BaseEstimator):
    """
    Some sklearn classifiers, such as logistic regression, have issues with datasets
    that contain a single class. This meta-classifier fixes the issue. If the
    training data contains a single class, this meta-classifier always returns that
    class as a prediction. Otherwise, it fits the provided base classifier,
    and returns its predictions instead.
    """

    def __init__(
        self,
        base_clf: BaseEstimator,
        clone_fn: Callable = sklearn.base.clone,
    ):
        self.base_clf = base_clf
        self.clf_: Optional[BaseEstimator] = None
        self.constant_ = None
        self.classes_ = None
        self.clone_fn = clone_fn

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        classes = unique_labels(y)
        if len(classes) == 1:
            assert classes[0] is not None
            self.clf_ = None
            self.constant_ = classes[0]
            self.classes_ = classes
        else:
            self.clf_ = self.clone_fn(self.base_clf)
            assert self.clf_ is not None
            self.clf_.fit(x, y)
            self.constant_ = None
            self.classes_ = self.clf_.classes_

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(x.shape[0], self.constant_)
        else:
            assert self.clf_ is not None
            return self.clf_.predict(x)
