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
    def __init__(self,
                 use_lazy_callbacks=True,
                 options=None):
        """
        Creates a new Gurobi solver, accessed through Pyomo.

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
            try:
                # User cuts
                if cb_where == GRB.Callback.MIPNODE:
                    logger.debug("Finding violated cutting planes...")
                    cb_opt.cbGetNodeRel(self._all_vars)
                    violations = self.instance.find_violated_user_cuts(cb_model)
                    self.instance.found_violated_user_cuts += violations
                    logger.debug("    %d found" % len(violations))
                    for v in violations:
                        cut = self.instance.build_user_cut(cb_model, v)
                        cb_opt.cbCut(cut)

                # Lazy constraints
                if cb_where == GRB.Callback.MIPSOL:
                    cb_opt.cbGetSolution(self._all_vars)
                    logger.debug("Finding violated lazy constraints...")
                    violations = self.instance.find_violated_lazy_constraints(cb_model)
                    self.instance.found_violated_lazy_constraints += violations
                    logger.debug("    %d found" % len(violations))
                    for v in violations:
                        cut = self.instance.build_lazy_constraint(cb_model, v)
                        cb_opt.cbLazy(cut)
            except Exception as e:
                logger.error(e)

        self._pyomo_solver.options["LazyConstraints"] = 1
        self._pyomo_solver.options["PreCrush"] = 1
        self._pyomo_solver.set_callback(cb)

        self.instance.found_violated_lazy_constraints = []
        self.instance.found_violated_user_cuts = []

        streams = [StringIO()]
        if tee:
            streams += [sys.stdout]
        with RedirectOutput(streams):
            results = self._pyomo_solver.solve(tee=True,
                                               warmstart=self._is_warm_start_available)

        self._pyomo_solver.set_callback(None)
        log = streams[0].getvalue()
        return {
            "Lower bound": results["Problem"][0]["Lower bound"],
            "Upper bound": results["Problem"][0]["Upper bound"],
            "Wallclock time": results["Solver"][0]["Wallclock time"],
            "Nodes": self._extract_node_count(log),
            "Sense": self._obj_sense,
            "Log": log,
            "Warm start value": self._extract_warm_start_value(log),
        }

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
