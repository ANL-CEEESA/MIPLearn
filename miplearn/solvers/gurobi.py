#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
import re
import sys
from io import StringIO
from random import randint
from typing import List, Any, Dict, Optional, cast, Tuple, Union

from miplearn.instance import Instance
from miplearn.solvers import _RedirectOutput
from miplearn.solvers.internal import (
    InternalSolver,
    LPSolveStats,
    IterationCallback,
    LazyCallback,
    MIPSolveStats,
)
from miplearn.types import VarIndex, SolverParams, Solution

logger = logging.getLogger(__name__)


class GurobiSolver(InternalSolver):
    """
    An InternalSolver backed by Gurobi's Python API (without Pyomo).

    Parameters
    ----------
    params: Optional[SolverParams]
        Parameters to pass to Gurobi. For example, `params={"MIPGap": 1e-3}`
        sets the gap tolerance to 1e-3.
    lazy_cb_frequency: int
        If 1, calls lazy constraint callbacks whenever an integer solution
        is found. If 2, calls it also at every node, after solving the
        LP relaxation of that node.
    """

    def __init__(
        self,
        params: Optional[SolverParams] = None,
        lazy_cb_frequency: int = 1,
    ) -> None:
        import gurobipy

        if params is None:
            params = {}
        params["InfUnbdInfo"] = True

        self.gp = gurobipy
        self.instance: Optional[Instance] = None
        self.model: Optional["gurobipy.Model"] = None
        self.params: SolverParams = params
        self._all_vars: Dict = {}
        self._bin_vars: Optional[Dict[str, Dict[VarIndex, "gurobipy.Var"]]] = None
        self.cb_where: Optional[int] = None

        assert lazy_cb_frequency in [1, 2]
        if lazy_cb_frequency == 1:
            self.lazy_cb_where = [self.gp.GRB.Callback.MIPSOL]
        else:
            self.lazy_cb_where = [
                self.gp.GRB.Callback.MIPSOL,
                self.gp.GRB.Callback.MIPNODE,
            ]

    def set_instance(
        self,
        instance: Instance,
        model: Any = None,
    ) -> None:
        self._raise_if_callback()
        if model is None:
            model = instance.to_model()
        assert isinstance(model, self.gp.Model)
        self.instance = instance
        self.model = model
        self.model.update()
        self._update_vars()

    def _raise_if_callback(self) -> None:
        if self.cb_where is not None:
            raise Exception("method cannot be called from a callback")

    def _update_vars(self) -> None:
        assert self.model is not None
        self._all_vars = {}
        self._bin_vars = {}
        idx: VarIndex
        for var in self.model.getVars():
            m = re.search(r"([^[]*)\[(.*)]", var.varName)
            if m is None:
                name = var.varName
                idx = (0,)
            else:
                name = m.group(1)
                parts = m.group(2).split(",")
                idx = cast(
                    Tuple[Union[str, int]],
                    tuple(int(k) if k.isdecimal() else str(k) for k in parts),
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

    def _apply_params(self, streams: List[Any]) -> None:
        assert self.model is not None
        with _RedirectOutput(streams):
            for (name, value) in self.params.items():
                self.model.setParam(name, value)
            if "seed" not in [k.lower() for k in self.params.keys()]:
                self.model.setParam("Seed", randint(0, 1_000_000))

    def solve_lp(
        self,
        tee: bool = False,
    ) -> LPSolveStats:
        self._raise_if_callback()
        streams: List[Any] = [StringIO()]
        if tee:
            streams += [sys.stdout]
        self._apply_params(streams)
        assert self.model is not None
        assert self._bin_vars is not None
        for (varname, vardict) in self._bin_vars.items():
            for (idx, var) in vardict.items():
                var.vtype = self.gp.GRB.CONTINUOUS
                var.lb = 0.0
                var.ub = 1.0
        with _RedirectOutput(streams):
            self.model.optimize()
        for (varname, vardict) in self._bin_vars.items():
            for (idx, var) in vardict.items():
                var.vtype = self.gp.GRB.BINARY
        log = streams[0].getvalue()
        opt_value = None
        if not self.is_infeasible():
            opt_value = self.model.objVal
        return {
            "Optimal value": opt_value,
            "Log": log,
        }

    def solve(
        self,
        tee: bool = False,
        iteration_cb: IterationCallback = None,
        lazy_cb: LazyCallback = None,
    ) -> MIPSolveStats:
        self._raise_if_callback()
        assert self.model is not None

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
        streams: List[Any] = [StringIO()]
        if tee:
            streams += [sys.stdout]
        self._apply_params(streams)
        if iteration_cb is None:
            iteration_cb = lambda: False
        while True:
            with _RedirectOutput(streams):
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
        ub, lb = None, None
        sense = "min" if self.model.modelSense == 1 else "max"
        if self.model.solCount > 0:
            if self.model.modelSense == 1:
                lb = self.model.objBound
                ub = self.model.objVal
            else:
                lb = self.model.objVal
                ub = self.model.objBound
        ws_value = self._extract_warm_start_value(log)
        stats: MIPSolveStats = {
            "Lower bound": lb,
            "Upper bound": ub,
            "Wallclock time": total_wallclock_time,
            "Nodes": total_nodes,
            "Sense": sense,
            "Log": log,
            "Warm start value": ws_value,
            "LP value": None,
        }
        return stats

    def get_solution(self) -> Optional[Solution]:
        self._raise_if_callback()
        assert self.model is not None
        if self.model.solCount == 0:
            return None
        solution: Solution = {}
        for (varname, vardict) in self._all_vars.items():
            solution[varname] = {}
            for (idx, var) in vardict.items():
                solution[varname][idx] = var.x
        return solution

    def set_warm_start(self, solution: Solution) -> None:
        self._raise_if_callback()
        self._clear_warm_start()
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

    def get_sense(self) -> str:
        assert self.model is not None
        if self.model.modelSense == 1:
            return "min"
        else:
            return "max"

    def get_value(
        self,
        var_name: str,
        index: VarIndex,
    ) -> Optional[float]:
        var = self._all_vars[var_name][index]
        return self._get_value(var)

    def is_infeasible(self) -> bool:
        assert self.model is not None
        return self.model.status in [self.gp.GRB.INFEASIBLE, self.gp.GRB.INF_OR_UNBD]

    def get_dual(self, cid: str) -> float:
        assert self.model is not None
        c = self.model.getConstrByName(cid)
        if self.is_infeasible():
            return c.farkasDual
        else:
            return c.pi

    def _get_value(self, var: Any) -> Optional[float]:
        assert self.model is not None
        if self.cb_where == self.gp.GRB.Callback.MIPSOL:
            return self.model.cbGetSolution(var)
        elif self.cb_where == self.gp.GRB.Callback.MIPNODE:
            return self.model.cbGetNodeRel(var)
        elif self.cb_where is None:
            if self.is_infeasible():
                return None
            else:
                return var.x
        else:
            raise Exception(
                "get_value cannot be called from cb_where=%s" % self.cb_where
            )

    def get_empty_solution(self) -> Solution:
        self._raise_if_callback()
        solution: Solution = {}
        for (varname, vardict) in self._all_vars.items():
            solution[varname] = {}
            for (idx, var) in vardict.items():
                solution[varname][idx] = None
        return solution

    def add_constraint(
        self,
        constraint: Any,
        name: str = "",
    ) -> None:
        assert self.model is not None
        if type(constraint) is tuple:
            lhs, sense, rhs, name = constraint
            if self.cb_where in [
                self.gp.GRB.Callback.MIPSOL,
                self.gp.GRB.Callback.MIPNODE,
            ]:
                self.model.cbLazy(lhs, sense, rhs)
            else:
                self.model.addConstr(lhs, sense, rhs, name)
        else:
            if self.cb_where in [
                self.gp.GRB.Callback.MIPSOL,
                self.gp.GRB.Callback.MIPNODE,
            ]:
                self.model.cbLazy(constraint)
            else:
                self.model.addConstr(constraint, name=name)

    def _clear_warm_start(self) -> None:
        for (varname, vardict) in self._all_vars.items():
            for (idx, var) in vardict.items():
                var.start = self.gp.GRB.UNDEFINED

    def fix(self, solution: Solution) -> None:
        self._raise_if_callback()
        for (varname, vardict) in solution.items():
            for (idx, value) in vardict.items():
                if value is None:
                    continue
                var = self._all_vars[varname][idx]
                var.vtype = self.gp.GRB.CONTINUOUS
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

    def is_constraint_satisfied(self, cobj, tol=1e-6):
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

    def get_inequality_slacks(self) -> Dict[str, float]:
        assert self.model is not None
        ineqs = [c for c in self.model.getConstrs() if c.sense != "="]
        return {c.ConstrName: c.Slack for c in ineqs}

    def set_constraint_sense(self, cid: str, sense: str) -> None:
        assert self.model is not None
        c = self.model.getConstrByName(cid)
        c.Sense = sense

    def get_constraint_sense(self, cid: str) -> str:
        assert self.model is not None
        c = self.model.getConstrByName(cid)
        return c.Sense

    def relax(self) -> None:
        assert self.model is not None
        self.model.update()
        self.model = self.model.relax()
        self._update_vars()

    def _extract_warm_start_value(self, log: str) -> Optional[float]:
        ws = self.__extract(log, "MIP start with objective ([0-9.e+-]*)")
        if ws is None:
            return None
        return float(ws)

    @staticmethod
    def __extract(
        log: str,
        regexp: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
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

        self.params = state["params"]
        self.lazy_cb_where = state["lazy_cb_where"]
        self.instance = None
        self.model = None
        self._all_vars = None
        self._bin_vars = None
        self.cb_where = None
