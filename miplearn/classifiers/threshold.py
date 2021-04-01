#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import abstractmethod, ABC
from typing import Optional, List

import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve

from miplearn.classifiers import Classifier


class Threshold(ABC):
    """
    Solver components ask the machine learning models how confident are they on each
    prediction they make, then automatically discard all predictions that have low
    confidence. A Threshold specifies how confident should the ML models be for a
    prediction to be considered trustworthy.

    To model dynamic thresholds, which automatically adjust themselves during
    training to reach some desired target (such as minimum precision, or minimum
    recall), thresholds behave somewhat similar to ML models themselves, with `fit`
    and `predict` methods.
    """

    @abstractmethod
    def fit(
        self,
        clf: Classifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Given a trained binary classifier `clf`, calibrates itself based on the
        classifier's performance on the given training data set.
        """
        assert isinstance(clf, Classifier)
        assert isinstance(x_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        n_samples = x_train.shape[0]
        assert y_train.shape[0] == n_samples

    @abstractmethod
    def predict(self, x_test: np.ndarray) -> List[float]:
        """
        Returns the minimum probability for a machine learning prediction to be
        considered trustworthy. There is one value for each label.
        """
        pass

    @abstractmethod
    def clone(self) -> "Threshold":
        """
        Returns an unfitted copy of this threshold with the same hyperparameters.
        """
        pass


class MinProbabilityThreshold(Threshold):
    """
    A threshold which considers predictions trustworthy if their probability of being
    correct, as computed by the machine learning models, are above a fixed value.
    """

    def __init__(self, min_probability: List[float]):
        self.min_probability = min_probability

    def fit(self, clf: Classifier, x_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def predict(self, x_test: np.ndarray) -> List[float]:
        return self.min_probability

    def clone(self) -> "MinProbabilityThreshold":
        return MinProbabilityThreshold(self.min_probability)


class MinPrecisionThreshold(Threshold):
    """
    A dynamic threshold which automatically adjusts itself during training to ensure
    that the component achieves at least a given precision `p` on the training data
    set. Note that increasing a component's minimum precision may reduce its recall.
    """

    def __init__(self, min_precision: List[float]) -> None:
        self.min_precision = min_precision
        self._computed_threshold: Optional[List[float]] = None

    def fit(
        self,
        clf: Classifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        super().fit(clf, x_train, y_train)
        (n_samples, n_classes) = y_train.shape
        proba = clf.predict_proba(x_train)
        self._computed_threshold = [
            self._compute(
                y_train[:, i],
                proba[:, i],
                self.min_precision[i],
            )
            for i in range(n_classes)
        ]

    def predict(self, x_test: np.ndarray) -> List[float]:
        assert self._computed_threshold is not None
        return self._computed_threshold

    @staticmethod
    def _compute(
        y_actual: np.ndarray,
        y_prob: np.ndarray,
        min_precision: float,
    ) -> float:
        fps, tps, thresholds = _binary_clf_curve(y_actual, y_prob)
        precision = tps / (tps + fps)
        for k in reversed(range(len(precision))):
            if precision[k] >= min_precision:
                return thresholds[k]
        return float("inf")

    def clone(self) -> "MinPrecisionThreshold":
        return MinPrecisionThreshold(
            min_precision=self.min_precision,
        )
