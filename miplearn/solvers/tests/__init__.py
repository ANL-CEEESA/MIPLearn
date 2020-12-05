#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from inspect import isclass
from miplearn import BasePyomoSolver, GurobiSolver, GurobiPyomoSolver
from miplearn.problems.knapsack import KnapsackInstance, GurobiKnapsackInstance


def _get_instance(solver):
    def _is_subclass_or_instance(solver, parentClass):
        return isinstance(solver, parentClass) or (
            isclass(solver) and issubclass(solver, parentClass)
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


def _get_internal_solvers():
    return [GurobiPyomoSolver, GurobiSolver]
