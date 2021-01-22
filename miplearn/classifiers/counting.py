#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Optional, cast

import numpy as np

from miplearn.classifiers import Classifier


class CountingClassifier(Classifier):
    """

    A classifier that generates constant predictions, based only on the frequency of
    the training labels. For example, suppose `y_train` is given by:
    ```python
    y_train = np.array([
        [True, False],
        [False, True],
        [False, True],
    ])
    ```
    Then `predict_proba` always returns `[0.33 0.66]` for every sample, regardless of
    `x_train`. It essentially counts how many times each label appeared, hence the name.

    """

    def __init__(self) -> None:
        super().__init__()
        self.mean: Optional[np.ndarray] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        super().fit(x_train, y_train)
        self.mean = cast(np.ndarray, np.mean(y_train, axis=0))

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        super().predict_proba(x_test)
        n_samples = x_test.shape[0]
        return np.array([self.mean for _ in range(n_samples)])

    def __repr__(self):
        return "CountingClassifier(mean=%s)" % self.mean
