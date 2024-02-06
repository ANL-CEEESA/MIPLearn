#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import List, Any, Callable, Optional

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels


class MinProbabilityClassifier(BaseEstimator):
    """
    Meta-classifier that returns NaN for predictions made by a base classifier that
    have probability below a given threshold. More specifically, this meta-classifier
    calls base_clf.predict_proba and compares the result against the provided
    thresholds. If the probability for one of the classes is above its threshold,
    the meta-classifier returns that prediction. Otherwise, it returns NaN.
    """

    def __init__(
        self,
        base_clf: Any,
        thresholds: List[float],
        clone_fn: Callable[[Any], Any] = sklearn.base.clone,
    ) -> None:
        assert len(thresholds) == 2
        self.base_clf = base_clf
        self.thresholds = thresholds
        self.clone_fn = clone_fn
        self.clf_: Optional[Any] = None
        self.classes_: Optional[List[Any]] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        assert len(y.shape) == 1
        assert len(x.shape) == 2
        classes = unique_labels(y)
        assert len(classes) == len(self.thresholds)

        self.clf_ = self.clone_fn(self.base_clf)
        self.clf_.fit(x, y)
        self.classes_ = self.clf_.classes_

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.clf_ is not None
        assert self.classes_ is not None

        y_proba = self.clf_.predict_proba(x)
        assert len(y_proba.shape) == 2
        assert y_proba.shape[0] == x.shape[0]
        assert y_proba.shape[1] == 2
        n_samples = x.shape[0]

        y_pred = []
        for sample_idx in range(n_samples):
            yi = float("nan")
            for class_idx, class_val in enumerate(self.classes_):
                if y_proba[sample_idx, class_idx] >= self.thresholds[class_idx]:
                    yi = class_val
            y_pred.append(yi)
        return np.array(y_pred)
