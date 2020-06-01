#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import numpy as np
import pyomo.environ as pe

from miplearn import Instance, GurobiSolver, LearningSolver
from miplearn.problems.knapsack import ChallengeA


class CutInstance(Instance):
    def to_model(self):
        model = pe.ConcreteModel()
        model.x = x = pe.Var([0, 1], domain=pe.Binary)
        model.OBJ = pe.Objective(expr=x[0] + x[1], sense=pe.maximize)
        model.eq = pe.Constraint(expr=2 * x[0] + 2 * x[1] <= 3)
        return model

    def get_instance_features(self):
        return np.zeros(0)

    def get_variable_features(self, var, index):
        return np.zeros(0)


def test_cut():
    challenge = ChallengeA()
    gurobi = GurobiSolver()
    solver = LearningSolver(solver=gurobi, time_limit=10)
    solver.solve(challenge.training_instances[0])
    # assert False
