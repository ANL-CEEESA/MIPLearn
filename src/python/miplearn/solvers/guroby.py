#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import re
import sys
import logging
from io import StringIO

from . import RedirectOutput
from .internal import InternalSolver

logger = logging.getLogger(__name__)


class GurobiSolver(InternalSolver):
    def __init__(self, params=None):
        if params is None:
            params = {
                "LazyConstraints": 1,
                "PreCrush": 1,
            }
        from gurobipy import GRB
        self.GRB = GRB
        self.instance = None
        self.model = None
        self.params = params
        self._all_vars = None
        self._bin_vars = None
        self._varname_to_var = None

    def set_instance(self, instance, model=None):
        if model is None:
            model = instance.to_model()
        self.instance = instance
        self.model = model
        self.model.update()
        self._update_vars()

    def _update_vars(self):
        self._all_vars = {}
        self._bin_vars = {}
        for var in self.model.getVars():
            m = re.search(r"([^[]*)\[(.*)\]", var.varName)
            if m is None:
                name = var.varName
                idx = [0]
            else:
                name = m.group(1)
                idx = tuple(int(k) if k.isdecimal() else k
                            for k in m.group(2).split(","))
            if len(idx) == 1:
                idx = idx[0]
            if name not in self._all_vars:
                self._all_vars[name] = {}
            self._all_vars[name][idx] = var
            if var.vtype != 'C':
                if name not in self._bin_vars:
                    self._bin_vars[name] = {}
                self._bin_vars[name][idx] = var

    def _apply_params(self):
        for (name, value) in self.params.items():
            self.model.setParam(name, value)

    def solve_lp(self, tee=False):
        self._apply_params()
        streams = [StringIO()]
        if tee:
            streams += [sys.stdout]
        for (varname, vardict) in self._bin_vars.items():
            for (idx, var) in vardict.items():
                var.vtype = self.GRB.CONTINUOUS
                var.lb = 0.0
                var.ub = 1.0
        with RedirectOutput(streams):
            self.model.optimize()
        for (varname, vardict) in self._bin_vars.items():
            for (idx, var) in vardict.items():
                var.vtype = self.GRB.BINARY
        log = streams[0].getvalue()
        return {
            "Optimal value": self.model.objVal,
            "Log": log
        }

    def solve(self, tee=False):
        all_vars = self.model.getVars()
        self.instance.found_violated_lazy_constraints = []
        self.instance.found_violated_user_cuts = []
        streams = [StringIO()]
        if tee:
            streams += [sys.stdout]

        def cb(cb_model, cb_where):
            try:
                # User cuts
                if cb_where == self.GRB.Callback.MIPNODE:
                    logger.debug("Finding violated cutting planes...")
                    violations = self.instance.find_violated_user_cuts(cb_model)
                    self.instance.found_violated_user_cuts += violations
                    logger.debug("    %d found" % len(violations))
                    for v in violations:
                        cut = self.instance.build_user_cut(cb_model, v)
                        cb_model.cbCut(cut)

                # Lazy constraints
                if cb_where == self.GRB.Callback.MIPSOL:
                    logger.debug("Finding violated lazy constraints...")
                    violations = self.instance.find_violated_lazy_constraints(cb_model)
                    self.instance.found_violated_lazy_constraints += violations
                    logger.debug("    %d found" % len(violations))
                    for v in violations:
                        cut = self.instance.build_lazy_constraint(cb_model, v)
                        cb_model.cbLazy(cut)
            except Exception as e:
                logger.error(e)

        with RedirectOutput(streams):
            self.model.optimize(cb)
        log = streams[0].getvalue()
        return {
            "Lower bound": self.model.objVal,
            "Upper bound": self.model.objBound,
            "Wallclock time": self.model.runtime,
            "Nodes": int(self.model.nodeCount),
            "Sense": ("min" if self.model.modelSense == 1 else "max"),
            "Log": log,
            "Warm start value": self._extract_warm_start_value(log),
        }

    def get_solution(self):
        solution = {}
        for (varname, vardict) in self._all_vars.items():
            solution[varname] = {}
            for (idx, var) in vardict.items():
                solution[varname][idx] = var.x
        return solution

    def add_constraint(self, constraint):
        self.model.addConstr(constraint)

    def set_warm_start(self, solution):
        count_fixed, count_total = 0, 0
        for (varname, vardict) in solution.items():
            for (idx, value) in vardict.items():
                count_total += 1
                if value is not None:
                    count_fixed += 1
                    self._all_vars[varname][idx].start = value
        logger.info("Setting start values for %d variables (out of %d)" %
                    (count_fixed, count_total))

    def clear_warm_start(self):
        for (varname, vardict) in self._all_vars:
            for (idx, var) in vardict.items():
                var[idx].start = self.GRB.UNDEFINED

    def fix(self, solution):
        for (varname, vardict) in solution.items():
            for (idx, value) in vardict.items():
                if value is None:
                    continue
                var = self._all_vars[varname][idx]
                var.vtype = self.GRB.CONTINUOUS
                var.lb = value
                var.ub = value

    def set_branching_priorities(self, priorities):
        logger.warning("set_branching_priorities not implemented")

    def set_threads(self, threads):
        self.params["Threads"] = threads

    def set_time_limit(self, time_limit):
        self.params["TimeLimit"] = time_limit

    def set_node_limit(self, node_limit):
        self.params["NodeLimit"] = node_limit

    def set_gap_tolerance(self, gap_tolerance):
        self.params["MIPGap"] = gap_tolerance

    def _extract_warm_start_value(self, log):
        ws = self.__extract(log, "MIP start with objective ([0-9.e+-]*)")
        if ws is not None:
            ws = float(ws)
        return ws

    def __extract(self, log, regexp, default=None):
        value = default
        for line in log.splitlines():
            matches = re.findall(regexp, line)
            if len(matches) == 0:
                continue
            value = matches[0]
        return value

    def __getstate__(self):
        return self.params

    def __setstate__(self, state):
        self.params = state
