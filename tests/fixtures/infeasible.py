#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Any

from pyomo import environ as pe

from miplearn.instance.base import Instance
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.pyomo.base import BasePyomoSolver
from tests.solvers import _is_subclass_or_instance


class InfeasiblePyomoInstance(Instance):
    def to_model(self) -> pe.ConcreteModel:
        model = pe.ConcreteModel()
        model.x = pe.Var([0], domain=pe.Binary)
        model.OBJ = pe.Objective(expr=model.x[0], sense=pe.maximize)
        model.eq = pe.Constraint(expr=model.x[0] >= 2)
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


def get_infeasible_instance(solver):
    if _is_subclass_or_instance(solver, BasePyomoSolver):
        return InfeasiblePyomoInstance()
    if _is_subclass_or_instance(solver, GurobiSolver):
        return InfeasibleGurobiInstance()
