#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging

from pyomo import environ as pe
from scipy.stats import randint

from miplearn.solvers.pyomo.base import BasePyomoSolver

logger = logging.getLogger(__name__)


class GurobiPyomoSolver(BasePyomoSolver):
    """
    An InternalSolver that uses Gurobi and the Pyomo modeling language.

    Parameters
    ----------
    params: dict
        Dictionary of options to pass to the Pyomo solver. For example,
        {"Threads": 4} to set the number of threads.
    """

    def __init__(self, params=None):
        super().__init__(
            solver_factory=pe.SolverFactory("gurobi_persistent"),
            params={
                "Seed": randint(low=0, high=1000).rvs(),
            },
        )

    def _extract_node_count(self, log):
        return max(1, int(self._pyomo_solver._solver_model.getAttr("NodeCount")))

    def _get_warm_start_regexp(self):
        return "MIP start with objective ([0-9.e+-]*)"

    def _get_node_count_regexp(self):
        return None

    def set_branching_priorities(self, priorities):
        from gurobipy import GRB

        for varname in priorities.keys():
            var = self._varname_to_var[varname]
            for (index, priority) in priorities[varname].items():
                gvar = self._pyomo_solver._pyomo_var_to_solver_var_map[var[index]]
                gvar.setAttr(GRB.Attr.BranchPriority, int(round(priority)))
