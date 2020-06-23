#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn import BasePyomoSolver, GurobiSolver, GurobiPyomoSolver, CplexPyomoSolver
from miplearn.problems.knapsack import KnapsackInstance, GurobiKnapsackInstance


def _get_instance(solver):
    if issubclass(solver, BasePyomoSolver):
        return KnapsackInstance(
            weights=[23., 26., 20., 18.],
            prices=[505., 352., 458., 220.],
            capacity=67.,
        )
    if issubclass(solver, GurobiSolver):
        return GurobiKnapsackInstance(
            weights=[23., 26., 20., 18.],
            prices=[505., 352., 458., 220.],
            capacity=67.,
        )
    assert False


def _get_internal_solvers():
    return [GurobiPyomoSolver, CplexPyomoSolver, GurobiSolver]
