#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from inspect import isclass
from typing import List, Callable

from miplearn.problems.knapsack import KnapsackInstance, GurobiKnapsackInstance
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.internal import InternalSolver
from miplearn.solvers.pyomo.base import BasePyomoSolver
from miplearn.solvers.pyomo.gurobi import GurobiPyomoSolver
from miplearn.solvers.pyomo.xpress import XpressPyomoSolver


def _get_instance(solver):
    def _is_subclass_or_instance(obj, parent_class):
        return isinstance(obj, parent_class) or (
            isclass(obj) and issubclass(obj, parent_class)
        )

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


def _get_internal_solvers() -> List[Callable[[], InternalSolver]]:
    return [GurobiPyomoSolver, GurobiSolver, XpressPyomoSolver]
