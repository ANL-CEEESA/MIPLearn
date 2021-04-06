#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
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
                [0.25, 0.75],
                [0.40, 0.60],
                [0.90, 0.10],
            ]
        )
    )
    x_train = np.array(
        [
            [0],
            [1],
            [2],
            [3],
        ]
    )
    y_train = np.array(
        [
            [False, True],
            [False, True],
            [True, False],
            [True, False],
        ]
    )

    threshold = MinPrecisionThreshold(min_precision=[1.0, 1.0])
    threshold.fit(clf, x_train, y_train)
    assert threshold.predict(x_train) == [0.40, 0.75]

    # threshold = MinPrecisionThreshold(min_precision=0.65)
    # threshold.fit(clf, x_train, y_train)
    # assert threshold.predict(x_train) == [0.0, 0.80]

    # threshold = MinPrecisionThreshold(min_precision=0.50)
    # threshold.fit(clf, x_train, y_train)
    # assert threshold.predict(x_train) == [0.0, 0.70]
    #
    # threshold = MinPrecisionThreshold(min_precision=0.00)
    # threshold.fit(clf, x_train, y_train)
    # assert threshold.predict(x_train) == [0.0, 0.70]
