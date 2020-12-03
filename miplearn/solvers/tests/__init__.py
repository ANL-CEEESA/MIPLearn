#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from inspect import isclass
from miplearn import BasePyomoSolver, GurobiSolver, GurobiPyomoSolver
from miplearn.problems.knapsack import KnapsackInstance, GurobiKnapsackInstance


def _get_instance(solver):
    def _is_subclass_or_instance(solver, parentClass):
        return isinstance(solver, parentClass) or (isclass(solver) and issubclass(solver, parentClass))

    if _is_subclass_or_instance(solver, BasePyomoSolver):
        return KnapsackInstance(
            weights=[23., 26., 20., 18.],
            prices=[505., 352., 458., 220.],
            capacity=67.,
        )

    if _is_subclass_or_instance(solver, GurobiSolver):
        return GurobiKnapsackInstance(
            weights=[23., 26., 20., 18.],
            prices=[505., 352., 458., 220.],
            capacity=67.,
        )

    assert False


def _get_internal_solvers():
    return [GurobiPyomoSolver, GurobiSolver]
