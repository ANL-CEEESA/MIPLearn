#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from unittest.mock import Mock

import numpy as np
from miplearn.classifiers import Classifier
from miplearn.classifiers.threshold import MinPrecisionThreshold


def test_threshold_dynamic():
    clf = Mock(spec=Classifier)
    clf.predict_proba = Mock(
        return_value=np.array(
            [
                [0.10, 0.90],
                [0.10, 0.90],
                [0.20, 0.80],
                [0.30, 0.70],
            ]
        )
    )
    x_train = np.array([0, 1, 2, 3])
    y_train = np.array([1, 1, 0, 0])

    threshold = MinPrecisionThreshold(min_precision=1.0)
    assert threshold.find(clf, x_train, y_train) == 0.90

    threshold = MinPrecisionThreshold(min_precision=0.65)
    assert threshold.find(clf, x_train, y_train) == 0.80

    threshold = MinPrecisionThreshold(min_precision=0.50)
    assert threshold.find(clf, x_train, y_train) == 0.70

    threshold = MinPrecisionThreshold(min_precision=0.00)
    assert threshold.find(clf, x_train, y_train) == 0.70
