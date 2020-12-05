#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import sys
import logging
from io import StringIO
from pyomo import environ as pe
from scipy.stats import randint

from .base import BasePyomoSolver
from .. import RedirectOutput

logger = logging.getLogger(__name__)


class GurobiPyomoSolver(BasePyomoSolver):
    def __init__(self, options=None):
        """
        Creates a new Gurobi solver, accessed through Pyomo.

        Parameters
        ----------
        options: dict
            Dictionary of options to pass to the Pyomo solver. For example,
            {"Threads": 4} to set the number of threads.
        """
        super().__init__()
        self._pyomo_solver = pe.SolverFactory("gurobi_persistent")
        self._pyomo_solver.options["Seed"] = randint(low=0, high=1000).rvs()
        if options is not None:
            for (key, value) in options.items():
                self._pyomo_solver.options[key] = value

    def _extract_node_count(self, log):
        return max(1, int(self._pyomo_solver._solver_model.getAttr("NodeCount")))

    def _get_warm_start_regexp(self):
        return "MIP start with objective ([0-9.e+-]*)"

    def _get_node_count_regexp(self):
        return None

    def _get_threads_option_name(self):
        return "Threads"

    def _get_time_limit_option_name(self):
        return "TimeLimit"

    def _get_node_limit_option_name(self):
        return "NodeLimit"

    def _get_gap_tolerance_option_name(self):
        return "MIPGap"

    def set_branching_priorities(self, priorities):
        from gurobipy import GRB

        for varname in priorities.keys():
            var = self._varname_to_var[varname]
            for (index, priority) in priorities[varname].items():
                gvar = self._pyomo_solver._pyomo_var_to_solver_var_map[var[index]]
                gvar.setAttr(GRB.Attr.BranchPriority, int(round(priority)))
