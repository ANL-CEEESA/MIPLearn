#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from math import log
from typing import List, Dict, Any
import numpy as np

import gurobipy as gp

from ..h5 import H5File


class ExpertBranchPriorityComponent:
    def __init__(self) -> None:
        pass

    def fit(self, train_h5: List[str]) -> None:
        pass

    def before_mip(self, test_h5: str, model: gp.Model, _: Dict[str, Any]) -> None:
        with H5File(test_h5, "r") as h5:
            var_names = h5.get_array("static_var_names")
            var_priority = h5.get_array("bb_var_priority")
            assert var_priority is not None
            assert var_names is not None

            for (var_idx, var_name) in enumerate(var_names):
                if np.isfinite(var_priority[var_idx]):
                    var = model.getVarByName(var_name.decode())
                    var.branchPriority = int(log(1 + var_priority[var_idx]))
