#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
import re
import sys
from dataclasses import dataclass
from io import StringIO
from random import randint
from typing import List, Any, Dict, Optional, Hashable

from overrides import overrides

from miplearn.features import Constraint
from miplearn.instance.base import Instance
from miplearn.solvers import _RedirectOutput
from miplearn.solvers.internal import (
    InternalSolver,
    LPSolveStats,
    IterationCallback,
    LazyCallback,
    MIPSolveStats,
)
from miplearn.solvers.pyomo.base import PyomoTestInstanceKnapsack
from miplearn.types import (
    SolverParams,
    UserCutCallback,
    Solution,
    VariableName,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedGurobiConstraint:
    lhs: Any
    rhs: float
    sense: str
    name: str


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
        iteration_cb: Optional[IterationCallback] = None,
        lazy_cb: Optional[LazyCallback] = None,
        user_cut_cb: Optional[UserCutCallback] = None,
    ) -> MIPSolveStats:
        self._raise_if_callback()
        assert self.model is not None
        if iteration_cb is None:
            iteration_cb = lambda: False
        callback_exceptions = []

        # Create callback wrapper
        def cb_wrapper(cb_model: Any, cb_where: int) -> None:
            try:
                self.cb_where = cb_where
                if lazy_cb is not None and cb_where in self.lazy_cb_where:
                    lazy_cb(self, self.model)
                if user_cut_cb is not None and cb_where == self.gp.GRB.Callback.MIPNODE:
                    user_cut_cb(self, self.model)
            except Exception as e:
                logger.exception("callback error")
                callback_exceptions.append(e)
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
            if len(callback_exceptions) > 0:
                raise callback_exceptions[0]
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
    def add_constraint(self, cobj: Any, name: str = "") -> None:
        assert self.model is not None
        if isinstance(cobj, ExtractedGurobiConstraint):
            if self.cb_where in [
                self.gp.GRB.Callback.MIPSOL,
                self.gp.GRB.Callback.MIPNODE,
            ]:
                self.model.cbLazy(cobj.lhs, cobj.sense, cobj.rhs)
            else:
                self.model.addConstr(cobj.lhs, cobj.sense, cobj.rhs, cobj.name)
        elif isinstance(cobj, self.gp.TempConstr):
            if self.cb_where in [
                self.gp.GRB.Callback.MIPSOL,
                self.gp.GRB.Callback.MIPNODE,
            ]:
                self.model.cbLazy(cobj)
            else:
                self.model.addConstr(cobj, name=name)
        else:
            raise Exception(f"unknown constraint type: {cobj.__class__.__name__}")

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
    def extract_constraint(self, cid: str) -> ExtractedGurobiConstraint:
        self._raise_if_callback()
        assert self.model is not None
        constr = self.model.getConstrByName(cid)
        cobj = ExtractedGurobiConstraint(
            lhs=self.model.getRow(constr),
            sense=constr.sense,
            rhs=constr.RHS,
            name=constr.ConstrName,
        )
        self.model.remove(constr)
        return cobj

    @overrides
    def is_constraint_satisfied(
        self,
        cobj: ExtractedGurobiConstraint,
        tol: float = 1e-6,
    ) -> bool:
        assert isinstance(cobj, ExtractedGurobiConstraint)
        lhs, sense, rhs, _ = cobj.lhs, cobj.sense, cobj.rhs, cobj.name
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

    def __getstate__(self) -> Dict:
        return {
            "params": self.params,
            "lazy_cb_where": self.lazy_cb_where,
        }

    def __setstate__(self, state: Dict) -> None:
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

    @overrides
    def build_test_instance_infeasible(self) -> Instance:
        return GurobiTestInstanceInfeasible()

    @overrides
    def build_test_instance_redundancy(self) -> Instance:
        return GurobiTestInstanceRedundancy()

    @overrides
    def build_test_instance_knapsack(self) -> Instance:
        return GurobiTestInstanceKnapsack(
            weights=[23.0, 26.0, 20.0, 18.0],
            prices=[505.0, 352.0, 458.0, 220.0],
            capacity=67.0,
        )

    @overrides
    def get_constraints(self) -> Dict[str, Constraint]:
        assert self.model is not None
        self._raise_if_callback()
        self.model.update()
        constraints: Dict[str, Constraint] = {}
        for c in self.model.getConstrs():
            constr = self._parse_gurobi_constraint(c)
            assert c.constrName not in constraints
            constraints[c.constrName] = constr
        return constraints

    def _parse_gurobi_constraint(self, c: Any) -> Constraint:
        assert self.model is not None
        expr = self.model.getRow(c)
        lhs: Dict[str, float] = {}
        for i in range(expr.size()):
            lhs[expr.getVar(i).varName] = expr.getCoeff(i)
        return Constraint(rhs=c.rhs, lhs=lhs, sense=c.sense)

    @overrides
    def are_callbacks_supported(self) -> bool:
        return True


class GurobiTestInstanceInfeasible(Instance):
    @overrides
    def to_model(self) -> Any:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model()
        x = model.addVars(1, vtype=GRB.BINARY, name="x")
        model.addConstr(x[0] >= 2)
        model.setObjective(x[0])
        return model


class GurobiTestInstanceRedundancy(Instance):
    def to_model(self) -> Any:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model()
        x = model.addVars(2, vtype=GRB.BINARY, name="x")
        model.addConstr(x[0] + x[1] <= 1)
        model.addConstr(x[0] + x[1] <= 2)
        model.setObjective(x[0] + x[1], GRB.MAXIMIZE)
        return model


class GurobiTestInstanceKnapsack(PyomoTestInstanceKnapsack):
    """
    Simpler (one-dimensional) knapsack instance, implemented directly in Gurobi
    instead of Pyomo, used for testing.
    """

    def __init__(
        self,
        weights: List[float],
        prices: List[float],
        capacity: float,
    ) -> None:
        super().__init__(weights, prices, capacity)

    @overrides
    def to_model(self) -> Any:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("Knapsack")
        n = len(self.weights)
        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        model.addConstr(
            gp.quicksum(x[i] * self.weights[i] for i in range(n)) <= self.capacity,
            "eq_capacity",
        )
        model.setObjective(
            gp.quicksum(x[i] * self.prices[i] for i in range(n)), GRB.MAXIMIZE
        )
        return model

    @overrides
    def build_lazy_constraint(self, model: Any, violation: Hashable) -> Any:
        # TODO: Replace by plain constraint
        return ExtractedGurobiConstraint(
            lhs=1.0 * model.getVarByName("x[0]"),
            sense="<",
            rhs=0.0,
            name="cut",
        )
