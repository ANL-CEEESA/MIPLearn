#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Any

from miplearn import Instance, BasePyomoSolver, GurobiSolver
import pyomo.environ as pe

from tests.solvers import _is_subclass_or_instance


class PyomoInstanceWithRedundancy(Instance):
    def to_model(self) -> pe.ConcreteModel:
        model = pe.ConcreteModel()
        model.x = pe.Var([0, 1], domain=pe.Binary)
        model.OBJ = pe.Objective(expr=model.x[0] + model.x[1], sense=pe.maximize)
        model.eq1 = pe.Constraint(expr=model.x[0] + model.x[1] <= 1)
        model.eq2 = pe.Constraint(expr=model.x[0] + model.x[1] <= 2)
        return model


class GurobiInstanceWithRedundancy(Instance):
    def to_model(self) -> Any:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model()
        x = model.addVars(2, vtype=GRB.BINARY, name="x")
        model.addConstr(x[0] + x[1] <= 1)
        model.addConstr(x[0] + x[1] <= 2)
        model.setObjective(x[0] + x[1], GRB.MAXIMIZE)
        return model


def get_instance_with_redundancy(solver):
    if _is_subclass_or_instance(solver, BasePyomoSolver):
        return PyomoInstanceWithRedundancy()
    if _is_subclass_or_instance(solver, GurobiSolver):
        return GurobiInstanceWithRedundancy()
