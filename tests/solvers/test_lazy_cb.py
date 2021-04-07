#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging

from miplearn.solvers.gurobi import GurobiSolver
from . import _get_knapsack_instance

logger = logging.getLogger(__name__)


def test_lazy_cb():
    solver = GurobiSolver()
    instance = _get_knapsack_instance(solver)
    model = instance.to_model()

    def lazy_cb(cb_solver, cb_model):
        cobj = (cb_model.getVarByName("x[0]") * 1.0, "<", 0.0, "cut")
        if not cb_solver.is_constraint_satisfied(cobj):
            cb_solver.add_constraint(cobj)

    solver.set_instance(instance, model)
    solver.solve(lazy_cb=lazy_cb)
    solution = solver.get_solution()
    assert solution["x[0]"] == 0.0
