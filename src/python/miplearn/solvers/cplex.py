#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import pyomo.environ as pe
from scipy.stats import randint

from .internal import InternalSolver


class CPLEXSolver(InternalSolver):
    def __init__(self, options=None):
        """
        Creates a new CPLEXSolver.

        Parameters
        ----------
        options: dict
            Dictionary of options to pass to the Pyomo solver. For example,
            {"mip_display": 5} to increase the log verbosity.
        """
        super().__init__()
        self._pyomo_solver = pe.SolverFactory('cplex_persistent')
        self._pyomo_solver.options["randomseed"] = randint(low=0, high=1000).rvs()
        self._pyomo_solver.options["mip_display"] = 4
        if options is not None:
            for (key, value) in options.items():
                self._pyomo_solver.options[key] = value

    def set_threads(self, threads):
        self._pyomo_solver.options["threads"] = threads

    def set_time_limit(self, time_limit):
        self._pyomo_solver.options["timelimit"] = time_limit

    def set_gap_tolerance(self, gap_tolerance):
        self._pyomo_solver.options["mip_tolerances_mipgap"] = gap_tolerance

    def solve_lp(self, tee=False):
        import cplex
        lp = self._pyomo_solver._solver_model
        var_types = lp.variables.get_types()
        n_vars = len(var_types)
        lp.set_problem_type(cplex.Cplex.problem_type.LP)
        results = self._pyomo_solver.solve(tee=tee)
        lp.variables.set_types(zip(range(n_vars), var_types))
        return {
            "Optimal value": results["Problem"][0]["Lower bound"],
        }

    def _get_warm_start_regexp(self):
        return "MIP start .* with objective ([0-9.e+-]*)\\."

    def _get_threads_option_name(self):
        return "threads"

    def _get_time_limit_option_name(self):
        return "timelimit"

    def _get_gap_tolerance_option_name(self):
        return "mip_gap_tolerances_mipgap"
