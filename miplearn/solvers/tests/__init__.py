#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from inspect import isclass
from typing import List, Callable, Any

from pyomo import environ as pe

from miplearn.instance import Instance, PyomoInstance
from miplearn.problems.knapsack import KnapsackInstance, GurobiKnapsackInstance
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.internal import InternalSolver
from miplearn.solvers.pyomo.base import BasePyomoSolver
from miplearn.solvers.pyomo.gurobi import GurobiPyomoSolver
from miplearn.solvers.pyomo.xpress import XpressPyomoSolver


class InfeasiblePyomoInstance(PyomoInstance):
    def to_model(self) -> pe.ConcreteModel:
        model = pe.ConcreteModel()
        model.x = pe.Var(domain=pe.Binary)
        model.OBJ = pe.Objective(expr=model.x, sense=pe.maximize)
        model.eq = pe.Constraint(expr=model.x >= 2)
        return model


class InfeasibleGurobiInstance(Instance):
    def to_model(self) -> Any:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model()
        x = model.addVars(1, vtype=GRB.BINARY, name="x")
        model.addConstr(x[0] >= 2)
        model.setObjective(x[0])
        return model


def _is_subclass_or_instance(obj, parent_class):
    return isinstance(obj, parent_class) or (
        isclass(obj) and issubclass(obj, parent_class)
    )


def _get_knapsack_instance(solver):
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


def _get_infeasible_instance(solver):
    if _is_subclass_or_instance(solver, BasePyomoSolver):
        return InfeasiblePyomoInstance()
    if _is_subclass_or_instance(solver, GurobiSolver):
        return InfeasibleGurobiInstance()


def _get_internal_solvers() -> List[Callable[[], InternalSolver]]:
    return [GurobiPyomoSolver, GurobiSolver, XpressPyomoSolver]
