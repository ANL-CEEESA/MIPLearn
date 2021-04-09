#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from inspect import isclass
from typing import Any

from miplearn.instance.base import Instance
from miplearn.problems.knapsack import KnapsackInstance, GurobiKnapsackInstance
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.pyomo.base import BasePyomoSolver


def _is_subclass_or_instance(obj: Any, parent_class: Any) -> bool:
    return isinstance(obj, parent_class) or (
        isclass(obj) and issubclass(obj, parent_class)
    )


def _get_knapsack_instance(solver: Any) -> Instance:
    if _is_subclass_or_instance(solver, BasePyomoSolver):
        return KnapsackInstance(
            weights=[23.0, 26.0, 20.0, 18.0],
            prices=[505.0, 352.0, 458.0, 220.0],
            capacity=67.0,
        )
    if _is_subclass_or_instance(solver, GurobiSolver):
        return GurobiKnapsackInstance(
            weights=[23.0, 26.0, 20.0, 18.0],
            prices=[505.0, 352.0, 458.0, 220.0],
            capacity=67.0,
        )
    assert False
