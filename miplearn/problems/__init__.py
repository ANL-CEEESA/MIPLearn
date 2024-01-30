#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Any, Optional

import gurobipy as gp
from pyomo import environ as pe


def _gurobipy_set_params(model: gp.Model, params: Optional[dict[str, Any]]) -> None:
    assert isinstance(model, gp.Model)
    if params is not None:
        for (param_name, param_value) in params.items():
            setattr(model.params, param_name, param_value)


def _pyomo_set_params(
    model: pe.ConcreteModel,
    params: Optional[dict[str, Any]],
    solver: str,
) -> None:
    assert (
        solver == "gurobi_persistent"
    ), "setting parameters is only supported with gurobi_persistent"
    if solver == "gurobi_persistent" and params is not None:
        for (param_name, param_value) in params.items():
            model.solver.set_gurobi_param(param_name, param_value)
