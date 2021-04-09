#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import List

import pytest

from miplearn import InternalSolver, GurobiPyomoSolver, GurobiSolver
from miplearn.solvers.pyomo.xpress import XpressPyomoSolver


@pytest.fixture
def internal_solvers() -> List[InternalSolver]:
    return [
        GurobiPyomoSolver(),
        GurobiSolver(),
        XpressPyomoSolver(),
    ]
