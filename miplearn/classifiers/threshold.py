#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import abstractmethod, ABC

import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve


class DynamicThreshold(ABC):
    @abstractmethod
    def find(self, clf, x_train, y_train):
        """
        Given a trained binary classifier `clf` and a training data set,
        returns the numerical threshold (float) satisfying some criterea.
        """
        pass


class MinPrecisionThreshold(DynamicThreshold):
    """
    The smallest possible threshold satisfying a minimum acceptable true
    positive rate (also known as precision).
    """

    def __init__(self, min_precision):
        self.min_precision = min_precision

    def find(self, clf, x_train, y_train):
        proba = clf.predict_proba(x_train)

        assert isinstance(proba, np.ndarray), "classifier should return numpy array"
        assert proba.shape == (
            x_train.shape[0],
            2,
        ), "classifier should return (%d,%d)-shaped array, not %s" % (
            x_train.shape[0],
            2,
            str(proba.shape),
        )

        fps, tps, thresholds = _binary_clf_curve(y_train, proba[:, 1])
        precision = tps / (tps + fps)

        for k in reversed(range(len(precision))):
            if precision[k] >= self.min_precision:
                return thresholds[k]
        return 2.0
