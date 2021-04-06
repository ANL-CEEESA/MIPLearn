#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def _build_circle_training_data() -> Tuple[np.ndarray, np.ndarray]:
    x_train = StandardScaler().fit_transform(
        np.array(
            [
                [
                    x1,
                    x2,
                ]
                for x1 in range(-10, 11)
                for x2 in range(-10, 11)
            ]
        )
    )
    y_train = np.array(
        [
            [
                False,
                True,
            ]
            if x1 * x1 + x2 * x2 <= 100
            else [
                True,
                False,
            ]
            for x1 in range(-10, 11)
            for x2 in range(-10, 11)
        ]
    )
    return x_train, y_train
