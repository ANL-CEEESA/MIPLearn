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
    def __init__(
        self,
        params=None,
        lazy_cb_frequency=1,
    ):
        """
        An InternalSolver backed by Gurobi's Python API (without Pyomo).

        Parameters
        ----------
        params
            Parameters to pass to Gurobi. For example, params={"MIPGap": 1e-3}
            sets the gap tolerance to 1e-3.
        lazy_cb_frequency
            If 1, calls lazy constraint callbacks whenever an integer solution
            is found. If 2, calls it also at every node, after solving the
            LP relaxation of that node.
        """
        if params is None:
            params = {}
        params["InfUnbdInfo"] = True
        from gurobipy import GRB

        self.GRB = GRB
        self.instance = None
        self.model = None
        self.params = params
        self._all_vars = None
        self._bin_vars = None
        self.cb_where = None
        assert lazy_cb_frequency in [1, 2]
        if lazy_cb_frequency == 1:
            self.lazy_cb_where = [self.GRB.Callback.MIPSOL]
        else:
            self.lazy_cb_where = [self.GRB.Callback.MIPSOL, self.GRB.Callback.MIPNODE]

    def set_instance(self, instance, model=None):
        self._raise_if_callback()
        if model is None:
            model = instance.to_model()
        self.instance = instance
        self.model = model
        self.model.update()
        self._update_vars()

    def _raise_if_callback(self):
        if self.cb_where is not None:
            raise Exception("method cannot be called from a callback")

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
                idx = tuple(
                    int(k) if k.isdecimal() else k for k in m.group(2).split(",")
                )
            if len(idx) == 1:
                idx = idx[0]
            if name not in self._all_vars:
                self._all_vars[name] = {}
            self._all_vars[name][idx] = var
            if var.vtype != "C":
                if name not in self._bin_vars:
                    self._bin_vars[name] = {}
                self._bin_vars[name][idx] = var

    def _apply_params(self, streams):
        with RedirectOutput(streams):
            for (name, value) in self.params.items():
                self.model.setParam(name, value)

    def solve_lp(self, tee=False):
        self._raise_if_callback()
        streams = [StringIO()]
        if tee:
            streams += [sys.stdout]
        self._apply_params(streams)
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
        return {"Optimal value": self.model.objVal, "Log": log}

    def solve(self, tee=False, iteration_cb=None, lazy_cb=None):
        self._raise_if_callback()

        def cb_wrapper(cb_model, cb_where):
            try:
                self.cb_where = cb_where
                if cb_where in self.lazy_cb_where:
                    lazy_cb(self, self.model)
            except:
                logger.exception("callback error")
            finally:
                self.cb_where = None

        if lazy_cb:
            self.params["LazyConstraints"] = 1
        total_wallclock_time = 0
        total_nodes = 0
        streams = [StringIO()]
        if tee:
            streams += [sys.stdout]
        self._apply_params(streams)
        if iteration_cb is None:
            iteration_cb = lambda: False
        while True:
            with RedirectOutput(streams):
                if lazy_cb is None:
                    self.model.optimize()
                else:
                    self.model.optimize(cb_wrapper)
            total_wallclock_time += self.model.runtime
            total_nodes += int(self.model.nodeCount)
            should_repeat = iteration_cb()
            if not should_repeat:
                break
        log = streams[0].getvalue()
        if self.model.modelSense == 1:
            sense = "min"
            lb = self.model.objBound
            ub = self.model.objVal
        else:
            sense = "max"
            lb = self.model.objVal
            ub = self.model.objBound
        return {
            "Lower bound": lb,
            "Upper bound": ub,
            "Wallclock time": total_wallclock_time,
            "Nodes": total_nodes,
            "Sense": sense,
            "Log": log,
            "Warm start value": self._extract_warm_start_value(log),
        }

    def get_solution(self):
        self._raise_if_callback()

        solution = {}
        for (varname, vardict) in self._all_vars.items():
            solution[varname] = {}
            for (idx, var) in vardict.items():
                solution[varname][idx] = var.x
        return solution

    def get_value(self, var_name, index):
        var = self._all_vars[var_name][index]
        return self._get_value(var)

    def is_infeasible(self):
        return self.model.status in [self.GRB.INFEASIBLE, self.GRB.INF_OR_UNBD]

    def get_farkas_dual(self, cid):
        c = self.model.getConstrByName(cid)
        return c.farkasDual

    def _get_value(self, var):
        if self.cb_where == self.GRB.Callback.MIPSOL:
            return self.model.cbGetSolution(var)
        elif self.cb_where == self.GRB.Callback.MIPNODE:
            return self.model.cbGetNodeRel(var)
        elif self.cb_where is None:
            return var.x
        else:
            raise Exception(
                "get_value cannot be called from cb_where=%s" % self.cb_where
            )

    def get_variables(self):
        self._raise_if_callback()
        variables = {}
        for (varname, vardict) in self._all_vars.items():
            variables[varname] = []
            for (idx, var) in vardict.items():
                variables[varname] += [idx]
        return variables

    def add_constraint(self, constraint, name=""):
        if type(constraint) is tuple:
            lhs, sense, rhs, name = constraint
            if self.cb_where in [self.GRB.Callback.MIPSOL, self.GRB.Callback.MIPNODE]:
                self.model.cbLazy(lhs, sense, rhs)
            else:
                self.model.addConstr(lhs, sense, rhs, name)
        else:
            if self.cb_where in [self.GRB.Callback.MIPSOL, self.GRB.Callback.MIPNODE]:
                self.model.cbLazy(constraint)
            else:
                self.model.addConstr(constraint, name=name)

    def set_warm_start(self, solution):
        self._raise_if_callback()
        count_fixed, count_total = 0, 0
        for (varname, vardict) in solution.items():
            for (idx, value) in vardict.items():
                count_total += 1
                if value is not None:
                    count_fixed += 1
                    self._all_vars[varname][idx].start = value
        logger.info(
            "Setting start values for %d variables (out of %d)"
            % (count_fixed, count_total)
        )

    def clear_warm_start(self):
        self._raise_if_callback()
        for (varname, vardict) in self._all_vars:
            for (idx, var) in vardict.items():
                var[idx].start = self.GRB.UNDEFINED

    def fix(self, solution):
        self._raise_if_callback()
        for (varname, vardict) in solution.items():
            for (idx, value) in vardict.items():
                if value is None:
                    continue
                var = self._all_vars[varname][idx]
                var.vtype = self.GRB.CONTINUOUS
                var.lb = value
                var.ub = value

    def get_constraint_ids(self):
        self._raise_if_callback()
        self.model.update()
        return [c.ConstrName for c in self.model.getConstrs()]

    def extract_constraint(self, cid):
        self._raise_if_callback()
        constr = self.model.getConstrByName(cid)
        cobj = (self.model.getRow(constr), constr.sense, constr.RHS, constr.ConstrName)
        self.model.remove(constr)
        return cobj

    def is_constraint_satisfied(self, cobj, tol=1e-5):
        lhs, sense, rhs, name = cobj
        if self.cb_where is not None:
            lhs_value = lhs.getConstant()
            for i in range(lhs.size()):
                var = lhs.getVar(i)
                coeff = lhs.getCoeff(i)
                lhs_value += self._get_value(var) * coeff
        else:
            lhs_value = lhs.getValue()
        if sense == "<":
            return lhs_value <= rhs + tol
        elif sense == ">":
            return lhs_value >= rhs - tol
        elif sense == "=":
            return abs(rhs - lhs_value) < abs(tol)
        else:
            raise Exception("Unknown sense: %s" % sense)

    def get_inequality_slacks(self):
        ineqs = [c for c in self.model.getConstrs() if c.sense != "="]
        return {c.ConstrName: c.Slack for c in ineqs}

    def set_constraint_sense(self, cid, sense):
        c = self.model.getConstrByName(cid)
        c.Sense = sense

    def get_constraint_sense(self, cid):
        c = self.model.getConstrByName(cid)
        return c.Sense

    def set_constraint_rhs(self, cid, rhs):
        c = self.model.getConstrByName(cid)
        c.RHS = rhs

    def relax(self):
        self.model = self.model.relax()
        self._update_vars()

    def set_branching_priorities(self, priorities):
        self._raise_if_callback()
        logger.warning("set_branching_priorities not implemented")

    def set_threads(self, threads):
        self._raise_if_callback()
        self.params["Threads"] = threads

    def set_time_limit(self, time_limit):
        self._raise_if_callback()
        self.params["TimeLimit"] = time_limit

    def set_node_limit(self, node_limit):
        self._raise_if_callback()
        self.params["NodeLimit"] = node_limit

    def set_gap_tolerance(self, gap_tolerance):
        self._raise_if_callback()
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
        return {
            "params": self.params,
            "lazy_cb_where": self.lazy_cb_where,
        }

    def __setstate__(self, state):
        from gurobipy import GRB

        self.params = state["params"]
        self.lazy_cb_where = state["lazy_cb_where"]
        self.GRB = GRB
        self.instance = None
        self.model = None
        self._all_vars = None
        self._bin_vars = None
        self.cb_where = None
