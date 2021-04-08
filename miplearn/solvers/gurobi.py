#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
import re
import sys
from io import StringIO
from random import randint
from typing import List, Any, Dict, Optional

from overrides import overrides

from miplearn.instance.base import Instance
from miplearn.solvers import _RedirectOutput
from miplearn.solvers.internal import (
    InternalSolver,
    LPSolveStats,
    IterationCallback,
    LazyCallback,
    MIPSolveStats,
)
from miplearn.types import (
    SolverParams,
    UserCutCallback,
    Solution,
    VariableName,
)

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

        assert lazy_cb_frequency in [1, 2]
        if params is None:
            params = {}
        params["InfUnbdInfo"] = True
        params["Seed"] = randint(0, 1_000_000)

        self.gp = gurobipy
        self.instance: Optional[Instance] = None
        self.model: Optional["gurobipy.Model"] = None
        self.params: SolverParams = params
        self.varname_to_var: Dict[str, "gurobipy.Var"] = {}
        self.bin_vars: List["gurobipy.Var"] = []
        self.cb_where: Optional[int] = None
        self.lazy_cb_frequency = lazy_cb_frequency

        if self.lazy_cb_frequency == 1:
            self.lazy_cb_where = [self.gp.GRB.Callback.MIPSOL]
        else:
            self.lazy_cb_where = [
                self.gp.GRB.Callback.MIPSOL,
                self.gp.GRB.Callback.MIPNODE,
            ]

    @overrides
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
        self.varname_to_var.clear()
        self.bin_vars.clear()
        for var in self.model.getVars():
            assert var.varName not in self.varname_to_var, (
                f"Duplicated variable name detected: {var.varName}. "
                f"Unique variable names are currently required."
            )
            self.varname_to_var[var.varName] = var
            assert var.vtype in ["B", "C"], (
                "Only binary and continuous variables are currently supported. "
                "Variable {var.varName} has type {var.vtype}."
            )
            if var.vtype == "B":
                self.bin_vars.append(var)

    def _apply_params(self, streams: List[Any]) -> None:
        assert self.model is not None
        with _RedirectOutput(streams):
            for (name, value) in self.params.items():
                self.model.setParam(name, value)

    @overrides
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
        for var in self.bin_vars:
            var.vtype = self.gp.GRB.CONTINUOUS
            var.lb = 0.0
            var.ub = 1.0
        with _RedirectOutput(streams):
            self.model.optimize()
        for var in self.bin_vars:
            var.vtype = self.gp.GRB.BINARY
        log = streams[0].getvalue()
        opt_value = None
        if not self.is_infeasible():
            opt_value = self.model.objVal
        return {
            "LP value": opt_value,
            "LP log": log,
        }

    @overrides
    def solve(
        self,
        tee: bool = False,
        iteration_cb: IterationCallback = None,
        lazy_cb: LazyCallback = None,
        user_cut_cb: UserCutCallback = None,
    ) -> MIPSolveStats:
        self._raise_if_callback()
        assert self.model is not None
        if iteration_cb is None:
            iteration_cb = lambda: False

        # Create callback wrapper
        def cb_wrapper(cb_model, cb_where):
            try:
                self.cb_where = cb_where
                if lazy_cb is not None and cb_where in self.lazy_cb_where:
                    lazy_cb(self, self.model)
                if user_cut_cb is not None and cb_where == self.gp.GRB.Callback.MIPNODE:
                    user_cut_cb(self, self.model)
            except:
                logger.exception("callback error")
            finally:
                self.cb_where = None

        # Configure Gurobi
        if lazy_cb is not None:
            self.params["LazyConstraints"] = 1
        if user_cut_cb is not None:
            self.params["PreCrush"] = 1

        # Solve problem
        total_wallclock_time = 0
        total_nodes = 0
        streams: List[Any] = [StringIO()]
        if tee:
            streams += [sys.stdout]
        self._apply_params(streams)
        while True:
            with _RedirectOutput(streams):
                self.model.optimize(cb_wrapper)
            total_wallclock_time += self.model.runtime
            total_nodes += int(self.model.nodeCount)
            should_repeat = iteration_cb()
            if not should_repeat:
                break

        # Fetch results and stats
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
            "MIP log": log,
            "Warm start value": ws_value,
        }
        return stats

    @overrides
    def get_solution(self) -> Optional[Solution]:
        self._raise_if_callback()
        assert self.model is not None
        if self.model.solCount == 0:
            return None
        return {v.varName: v.x for v in self.model.getVars()}

    @overrides
    def get_variable_names(self) -> List[VariableName]:
        self._raise_if_callback()
        assert self.model is not None
        return [v.varName for v in self.model.getVars()]

    @overrides
    def set_warm_start(self, solution: Solution) -> None:
        self._raise_if_callback()
        self._clear_warm_start()
        for (var_name, value) in solution.items():
            var = self.varname_to_var[var_name]
            if value is not None:
                var.start = value

    @overrides
    def get_sense(self) -> str:
        assert self.model is not None
        if self.model.modelSense == 1:
            return "min"
        else:
            return "max"

    @overrides
    def is_infeasible(self) -> bool:
        assert self.model is not None
        return self.model.status in [self.gp.GRB.INFEASIBLE, self.gp.GRB.INF_OR_UNBD]

    @overrides
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

    @overrides
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

    @overrides
    def add_cut(self, cobj: Any) -> None:
        assert self.model is not None
        assert self.cb_where == self.gp.GRB.Callback.MIPNODE
        self.model.cbCut(cobj)

    def _clear_warm_start(self) -> None:
        for var in self.varname_to_var.values():
            var.start = self.gp.GRB.UNDEFINED

    @overrides
    def fix(self, solution: Solution) -> None:
        self._raise_if_callback()
        for (varname, value) in solution.items():
            if value is None:
                continue
            var = self.varname_to_var[varname]
            var.vtype = self.gp.GRB.CONTINUOUS
            var.lb = value
            var.ub = value

    @overrides
    def get_constraint_ids(self):
        self._raise_if_callback()
        self.model.update()
        return [c.ConstrName for c in self.model.getConstrs()]

    @overrides
    def get_constraint_rhs(self, cid: str) -> float:
        assert self.model is not None
        return self.model.getConstrByName(cid).rhs

    @overrides
    def get_constraint_lhs(self, cid: str) -> Dict[str, float]:
        assert self.model is not None
        constr = self.model.getConstrByName(cid)
        expr = self.model.getRow(constr)
        lhs: Dict[str, float] = {}
        for i in range(expr.size()):
            lhs[expr.getVar(i).varName] = expr.getCoeff(i)
        return lhs

    @overrides
    def extract_constraint(self, cid):
        self._raise_if_callback()
        constr = self.model.getConstrByName(cid)
        cobj = (self.model.getRow(constr), constr.sense, constr.RHS, constr.ConstrName)
        self.model.remove(constr)
        return cobj

    @overrides
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

    @overrides
    def get_inequality_slacks(self) -> Dict[str, float]:
        assert self.model is not None
        ineqs = [c for c in self.model.getConstrs() if c.sense != "="]
        return {c.ConstrName: c.Slack for c in ineqs}

    @overrides
    def set_constraint_sense(self, cid: str, sense: str) -> None:
        assert self.model is not None
        c = self.model.getConstrByName(cid)
        c.Sense = sense

    @overrides
    def get_constraint_sense(self, cid: str) -> str:
        assert self.model is not None
        c = self.model.getConstrByName(cid)
        return c.Sense

    @overrides
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
        self.cb_where = None

    @overrides
    def clone(self) -> "GurobiSolver":
        return GurobiSolver(
            params=self.params,
            lazy_cb_frequency=self.lazy_cb_frequency,
        )
