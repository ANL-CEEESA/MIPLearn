#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Any, Dict, List

from . import _extract_bin_var_names_values
from .actions import PrimalComponentAction
from ...solvers.abstract import AbstractModel
from ...h5 import H5File

logger = logging.getLogger(__name__)


class ExpertPrimalComponent:
    def __init__(self, action: PrimalComponentAction):
        self.action = action

    """
    Component that predicts warm starts by peeking at the optimal solution.
    """

    def fit(self, train_h5: List[str]) -> None:
        pass

    def before_mip(
        self, test_h5: str, model: AbstractModel, stats: Dict[str, Any]
    ) -> None:
        with H5File(test_h5, "r") as h5:
            names, values, _ = _extract_bin_var_names_values(h5)
            self.action.perform(model, names, values.reshape(1, -1), stats)
