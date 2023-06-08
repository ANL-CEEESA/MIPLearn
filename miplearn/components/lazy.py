#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import json
from typing import Any, Dict, List

import gurobipy as gp

from ..h5 import H5File


class ExpertLazyComponent:
    def __init__(self) -> None:
        pass

    def fit(self, train_h5: List[str]) -> None:
        pass

    def before_mip(self, test_h5: str, model: gp.Model, stats: Dict[str, Any]) -> None:
        with H5File(test_h5, "r") as h5:
            constr_names = h5.get_array("static_constr_names")
            constr_lazy = h5.get_array("mip_constr_lazy")
            constr_violations = h5.get_scalar("mip_constr_violations")

            assert constr_names is not None
            assert constr_violations is not None

            # Static lazy constraints
            n_static_lazy = 0
            if constr_lazy is not None:
                for (constr_idx, constr_name) in enumerate(constr_names):
                    if constr_lazy[constr_idx]:
                        constr = model.getConstrByName(constr_name.decode())
                        constr.lazy = 3
                        n_static_lazy += 1
            stats.update({"Static lazy constraints": n_static_lazy})

            # Dynamic lazy constraints
            if hasattr(model, "_fix_violations"):
                violations = json.loads(constr_violations)
                model._fix_violations(model, violations, "aot")
                stats.update({"Dynamic lazy constraints": len(violations)})
