#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np

from miplearn.problems.setpack import (
    SetPackData,
    build_setpack_model_gurobipy,
)


def test_setpack() -> None:
    data = SetPackData(
        costs=np.array([5, 10, 12, 6, 8]),
        incidence_matrix=np.array(
            [
                [1, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1],
            ],
        ),
    )
    model = build_setpack_model_gurobipy(data)
    model.optimize()
    assert model.inner.objval == -22.0
