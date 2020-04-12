#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import sys
from io import StringIO

import pyomo.environ as pe
from miplearn.solvers import RedirectOutput
from miplearn.solvers.internal import InternalSolver
from scipy.stats import randint

logger = logging.getLogger(__name__)


class GurobiSolver(InternalSolver):
    def __init__(self,
                 use_lazy_callbacks=False,
                 options=None):
        """
        Creates a new GurobiSolver.

        Parameters
        ----------
        use_lazy_callbacks: bool
            If true, lazy constraints will be enforced via lazy callbacks.
            Otherwise, they will be enforced via a simple solve-check loop.
        options: dict
            Dictionary of options to pass to the Pyomo solver. For example,
            {"Threads": 4} to set the number of threads.
        """
        super().__init__()
        self._use_lazy_callbacks = use_lazy_callbacks
        self._pyomo_solver = pe.SolverFactory('gurobi_persistent')
        self._pyomo_solver.options["Seed"] = randint(low=0, high=1000).rvs()
        if options is not None:
            for (key, value) in options.items():
                self._pyomo_solver.options[key] = value

    def solve(self, tee=False):
        if self._use_lazy_callbacks:
            return self._solve_with_callbacks(tee)
        else:
            return super().solve(tee)

    def _solve_with_callbacks(self, tee):
        from gurobipy import GRB

        def cb(cb_model, cb_opt, cb_where):
            if cb_where == GRB.Callback.MIPSOL:
                cb_opt.cbGetSolution(self._all_vars)
                logger.debug("Finding violated constraints...")
                violations = self.instance.find_violations(cb_model)
                self.instance.found_violations += violations
                logger.debug("    %d violations found" % len(violations))
                for v in violations:
                    cut = self.instance.build_lazy_constraint(cb_model, v)
                    cb_opt.cbLazy(cut)

        if hasattr(self.instance, "find_violations"):
            self._pyomo_solver.options["LazyConstraints"] = 1
            self._pyomo_solver.set_callback(cb)
            self.instance.found_violations = []
        print(self._is_warm_start_available)
        streams = [StringIO()]
        if tee:
            streams += [sys.stdout]
        with RedirectOutput(streams):
            results = self._pyomo_solver.solve(tee=True,
                                               warmstart=self._is_warm_start_available)
        self._pyomo_solver.set_callback(None)
        node_count = int(self._pyomo_solver._solver_model.getAttr("NodeCount"))
        log = streams[0].getvalue()
        return {
            "Lower bound": results["Problem"][0]["Lower bound"],
            "Upper bound": results["Problem"][0]["Upper bound"],
            "Wallclock time": results["Solver"][0]["Wallclock time"],
            "Nodes": max(1, node_count),
            "Sense": self._obj_sense,
            "Log": log,
            "Warm start value": self.extract_warm_start_value(log),
        }

    def _get_warm_start_regexp(self):
        return "MIP start with objective ([0-9.e+-]*)"

    def _get_threads_option_name(self):
        return "Threads"

    def _get_time_limit_option_name(self):
        return "TimeLimit"

    def _get_gap_tolerance_option_name(self):
        return "MIPGap"
