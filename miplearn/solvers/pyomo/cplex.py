#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from pyomo import environ as pe
from scipy.stats import randint

from .base import BasePyomoSolver


class CplexPyomoSolver(BasePyomoSolver):
    def __init__(self, options=None):
        """
        Creates a new CPLEX solver, accessed through Pyomo.

        Parameters
        ----------
        options: dict
            Dictionary of options to pass to the Pyomo solver. For example,
            {"mip_display": 5} to increase the log verbosity.
        """
        super().__init__()
        self._pyomo_solver = pe.SolverFactory("cplex_persistent")
        self._pyomo_solver.options["randomseed"] = randint(low=0, high=1000).rvs()
        self._pyomo_solver.options["mip_display"] = 4
        if options is not None:
            for (key, value) in options.items():
                self._pyomo_solver.options[key] = value

    def _get_warm_start_regexp(self):
        return "MIP start .* with objective ([0-9.e+-]*)\\."

    def _get_node_count_regexp(self):
        return "^[ *] *([0-9]+)"

    def _get_threads_option_name(self):
        return "threads"

    def _get_time_limit_option_name(self):
        return "timelimit"

    def _get_node_limit_option_name(self):
        return "mip_limits_nodes"

    def _get_gap_tolerance_option_name(self):
        return "mip_tolerances_mipgap"

    def set_branching_priorities(self, priorities):
        raise NotImplementedError
