#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
import re
import sys
from io import StringIO
from random import randint
from typing import List, Any, Dict, Optional, Hashable, Tuple, TYPE_CHECKING

from overrides import overrides

from miplearn.features import VariableFeatures, ConstraintFeatures
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
)

if TYPE_CHECKING:
    import gurobipy

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
        self.cb_where: Optional[int] = None
        self.lazy_cb_frequency = lazy_cb_frequency
        self._dirty = True
        self._has_lp_solution = False
        self._has_mip_solution = False

        self._varname_to_var: Dict[str, "gurobipy.Var"] = {}
        self._cname_to_constr: Dict[str, "gurobipy.Constr"] = {}
        self._gp_vars: Tuple["gurobipy.Var", ...] = tuple()
        self._gp_constrs: Tuple["gurobipy.Constr", ...] = tuple()
        self._var_names: Tuple[str, ...] = tuple()
        self._constr_names: Tuple[str, ...] = tuple()
        self._var_types: Tuple[str, ...] = tuple()
        self._var_lbs: Tuple[float, ...] = tuple()
        self._var_ubs: Tuple[float, ...] = tuple()
        self._var_obj_coeffs: Tuple[float, ...] = tuple()

        if self.lazy_cb_frequency == 1:
            self.lazy_cb_where = [self.gp.GRB.Callback.MIPSOL]
        else:
            self.lazy_cb_where = [
                self.gp.GRB.Callback.MIPSOL,
                self.gp.GRB.Callback.MIPNODE,
            ]

    @overrides
    def add_constraints(self, cf: ConstraintFeatures) -> None:
        assert cf.names is not None
        assert cf.senses is not None
        assert cf.lhs is not None
        assert cf.rhs is not None
        assert self.model is not None
        for i in range(len(cf.names)):
            sense = cf.senses[i]
            lhs = self.gp.quicksum(
                self._varname_to_var[varname] * coeff for (varname, coeff) in cf.lhs[i]
            )
            if sense == "=":
                self.model.addConstr(lhs == cf.rhs[i], name=cf.names[i])
            elif sense == "<":
                self.model.addConstr(lhs <= cf.rhs[i], name=cf.names[i])
            else:
                self.model.addConstr(lhs >= cf.rhs[i], name=cf.names[i])
        self.model.update()
        self._dirty = True
        self._has_lp_solution = False
        self._has_mip_solution = False

    @overrides
    def are_callbacks_supported(self) -> bool:
        return True

    @overrides
    def are_constraints_satisfied(
        self,
        cf: ConstraintFeatures,
        tol: float = 1e-5,
    ) -> Tuple[bool, ...]:
        assert cf.names is not None
        assert cf.senses is not None
        assert cf.lhs is not None
        assert cf.rhs is not None
        assert self.model is not None
        result = []
        for i in range(len(cf.names)):
            sense = cf.senses[i]
            lhs = sum(
                self._varname_to_var[varname].x * coeff
                for (varname, coeff) in cf.lhs[i]
            )
            if sense == "<":
                result.append(lhs <= cf.rhs[i] + tol)
            elif sense == ">":
                result.append(lhs >= cf.rhs[i] - tol)
            else:
                result.append(abs(cf.rhs[i] - lhs) <= tol)
        return tuple(result)

    @overrides
    def build_test_instance_infeasible(self) -> Instance:
        return GurobiTestInstanceInfeasible()

    @overrides
    def build_test_instance_knapsack(self) -> Instance:
        return GurobiTestInstanceKnapsack(
            weights=[23.0, 26.0, 20.0, 18.0],
            prices=[505.0, 352.0, 458.0, 220.0],
            capacity=67.0,
        )

    @overrides
    def clone(self) -> "GurobiSolver":
        return GurobiSolver(
            params=self.params,
            lazy_cb_frequency=self.lazy_cb_frequency,
        )

    @overrides
    def fix(self, solution: Solution) -> None:
        self._raise_if_callback()
        for (varname, value) in solution.items():
            if value is None:
                continue
            var = self._varname_to_var[varname]
            var.vtype = self.gp.GRB.CONTINUOUS
            var.lb = value
            var.ub = value

    @overrides
    def get_constraint_attrs(self) -> List[str]:
        return [
            "basis_status",
            "categories",
            "dual_values",
            "lazy",
            "lhs",
            "names",
            "rhs",
            "sa_rhs_down",
            "sa_rhs_up",
            "senses",
            "slacks",
            "user_features",
        ]

    @overrides
    def get_constraints(
        self,
        with_static: bool = True,
        with_sa: bool = True,
        with_lhs: bool = True,
    ) -> ConstraintFeatures:
        model = self.model
        assert model is not None
        assert model.numVars == len(self._gp_vars)

        def _parse_gurobi_cbasis(v: int) -> str:
            if v == 0:
                return "B"
            if v == -1:
                return "N"
            raise Exception(f"unknown cbasis: {v}")

        gp_constrs = model.getConstrs()
        constr_names = tuple(model.getAttr("constrName", gp_constrs))
        rhs, lhs, senses, slacks, basis_status = None, None, None, None, None
        dual_value, basis_status, sa_rhs_up, sa_rhs_down = None, None, None, None

        if with_static:
            rhs = tuple(model.getAttr("rhs", gp_constrs))
            senses = tuple(model.getAttr("sense", gp_constrs))
            if with_lhs:
                lhs_l: List = [None for _ in gp_constrs]
                for (i, gp_constr) in enumerate(gp_constrs):
                    expr = model.getRow(gp_constr)
                    lhs_l[i] = tuple(
                        (self._var_names[expr.getVar(j).index], expr.getCoeff(j))
                        for j in range(expr.size())
                    )
                lhs = tuple(lhs_l)

        if self._has_lp_solution:
            dual_value = tuple(model.getAttr("pi", gp_constrs))
            basis_status = tuple(
                map(
                    _parse_gurobi_cbasis,
                    model.getAttr("cbasis", gp_constrs),
                )
            )
            if with_sa:
                sa_rhs_up = tuple(model.getAttr("saRhsUp", gp_constrs))
                sa_rhs_down = tuple(model.getAttr("saRhsLow", gp_constrs))

        if self._has_lp_solution or self._has_mip_solution:
            slacks = tuple(model.getAttr("slack", gp_constrs))

        return ConstraintFeatures(
            basis_status=basis_status,
            dual_values=dual_value,
            lhs=lhs,
            names=constr_names,
            rhs=rhs,
            sa_rhs_down=sa_rhs_down,
            sa_rhs_up=sa_rhs_up,
            senses=senses,
            slacks=slacks,
        )

    @overrides
    def get_solution(self) -> Optional[Solution]:
        assert self.model is not None
        if self.cb_where is not None:
            if self.cb_where == self.gp.GRB.Callback.MIPNODE:
                return {
                    v.varName: self.model.cbGetNodeRel(v) for v in self.model.getVars()
                }
            elif self.cb_where == self.gp.GRB.Callback.MIPSOL:
                return {
                    v.varName: self.model.cbGetSolution(v) for v in self.model.getVars()
                }
            else:
                raise Exception(
                    f"get_solution can only be called from a callback "
                    f"when cb_where is either MIPNODE or MIPSOL"
                )
        if self.model.solCount == 0:
            return None
        return {v.varName: v.x for v in self.model.getVars()}

    @overrides
    def get_variable_attrs(self) -> List[str]:
        return [
            "names",
            "basis_status",
            "categories",
            "lower_bounds",
            "obj_coeffs",
            "reduced_costs",
            "sa_lb_down",
            "sa_lb_up",
            "sa_obj_down",
            "sa_obj_up",
            "sa_ub_down",
            "sa_ub_up",
            "types",
            "upper_bounds",
            "user_features",
            "values",
        ]

    @overrides
    def get_variables(
        self,
        with_static: bool = True,
        with_sa: bool = True,
    ) -> VariableFeatures:
        model = self.model
        assert model is not None

        def _parse_gurobi_vbasis(b: int) -> str:
            if b == 0:
                return "B"
            elif b == -1:
                return "L"
            elif b == -2:
                return "U"
            elif b == -3:
                return "S"
            else:
                raise Exception(f"unknown vbasis: {basis_status}")

        upper_bounds, lower_bounds, types, values = None, None, None, None
        obj_coeffs, reduced_costs, basis_status = None, None, None
        sa_obj_up, sa_ub_up, sa_lb_up = None, None, None
        sa_obj_down, sa_ub_down, sa_lb_down = None, None, None

        if with_static:
            upper_bounds = self._var_ubs
            lower_bounds = self._var_lbs
            types = self._var_types
            obj_coeffs = self._var_obj_coeffs

        if self._has_lp_solution:
            reduced_costs = tuple(model.getAttr("rc", self._gp_vars))
            basis_status = tuple(
                map(
                    _parse_gurobi_vbasis,
                    model.getAttr("vbasis", self._gp_vars),
                )
            )

            if with_sa:
                sa_obj_up = tuple(model.getAttr("saobjUp", self._gp_vars))
                sa_obj_down = tuple(model.getAttr("saobjLow", self._gp_vars))
                sa_ub_up = tuple(model.getAttr("saubUp", self._gp_vars))
                sa_ub_down = tuple(model.getAttr("saubLow", self._gp_vars))
                sa_lb_up = tuple(model.getAttr("salbUp", self._gp_vars))
                sa_lb_down = tuple(model.getAttr("salbLow", self._gp_vars))

        if model.solCount > 0:
            values = tuple(model.getAttr("x", self._gp_vars))

        return VariableFeatures(
            names=self._var_names,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            types=types,
            obj_coeffs=obj_coeffs,
            reduced_costs=reduced_costs,
            basis_status=basis_status,
            sa_obj_up=sa_obj_up,
            sa_obj_down=sa_obj_down,
            sa_ub_up=sa_ub_up,
            sa_ub_down=sa_ub_down,
            sa_lb_up=sa_lb_up,
            sa_lb_down=sa_lb_down,
            values=values,
        )

    @overrides
    def is_infeasible(self) -> bool:
        assert self.model is not None
        return self.model.status in [self.gp.GRB.INFEASIBLE, self.gp.GRB.INF_OR_UNBD]

    @overrides
    def remove_constraints(self, names: Tuple[str, ...]) -> None:
        assert self.model is not None
        constrs = [self.model.getConstrByName(n) for n in names]
        self.model.remove(constrs)
        self.model.update()

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
        self._update()

    @overrides
    def set_warm_start(self, solution: Solution) -> None:
        self._raise_if_callback()
        self._clear_warm_start()
        for (var_name, value) in solution.items():
            var = self._varname_to_var[var_name]
            if value is not None:
                var.start = value

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
                self._dirty = False
            if len(callback_exceptions) > 0:
                raise callback_exceptions[0]
            total_wallclock_time += self.model.runtime
            total_nodes += int(self.model.nodeCount)
            should_repeat = iteration_cb()
            if not should_repeat:
                break
        self._has_lp_solution = False
        self._has_mip_solution = self.model.solCount > 0

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
        return MIPSolveStats(
            mip_lower_bound=lb,
            mip_upper_bound=ub,
            mip_wallclock_time=total_wallclock_time,
            mip_nodes=total_nodes,
            mip_sense=sense,
            mip_log=log,
            mip_warm_start_value=ws_value,
        )

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
        for (i, var) in enumerate(self._gp_vars):
            if self._var_types[i] == "B":
                var.vtype = self.gp.GRB.CONTINUOUS
                var.lb = 0.0
                var.ub = 1.0
        with _RedirectOutput(streams):
            self.model.optimize()
            self._dirty = False
        for (i, var) in enumerate(self._gp_vars):
            if self._var_types[i] == "B":
                var.vtype = self.gp.GRB.BINARY
        log = streams[0].getvalue()
        self._has_lp_solution = self.model.solCount > 0
        self._has_mip_solution = False
        opt_value = None
        if not self.is_infeasible():
            opt_value = self.model.objVal
        return LPSolveStats(
            lp_value=opt_value,
            lp_log=log,
            lp_wallclock_time=self.model.runtime,
        )

    def _apply_params(self, streams: List[Any]) -> None:
        assert self.model is not None
        with _RedirectOutput(streams):
            for (name, value) in self.params.items():
                self.model.setParam(name, value)

    def _clear_warm_start(self) -> None:
        for var in self._varname_to_var.values():
            var.start = self.gp.GRB.UNDEFINED

    @staticmethod
    def _extract(
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

    def _extract_warm_start_value(self, log: str) -> Optional[float]:
        ws = self._extract(log, "MIP start with objective ([0-9.e+-]*)")
        if ws is None:
            return None
        return float(ws)

    def _get_value(self, var: Any) -> float:
        assert self.model is not None
        if self.cb_where == self.gp.GRB.Callback.MIPSOL:
            return self.model.cbGetSolution(var)
        elif self.cb_where == self.gp.GRB.Callback.MIPNODE:
            return self.model.cbGetNodeRel(var)
        elif self.cb_where is None:
            return var.x
        else:
            raise Exception(
                "get_value cannot be called from cb_where=%s" % self.cb_where
            )

    def _raise_if_callback(self) -> None:
        if self.cb_where is not None:
            raise Exception("method cannot be called from a callback")

    def _update(self) -> None:
        assert self.model is not None
        gp_vars: List["gurobipy.Var"] = self.model.getVars()
        gp_constrs: List["gurobipy.Constr"] = self.model.getConstrs()
        var_names: List[str] = self.model.getAttr("varName", gp_vars)
        var_types: List[str] = self.model.getAttr("vtype", gp_vars)
        var_ubs: List[float] = self.model.getAttr("ub", gp_vars)
        var_lbs: List[float] = self.model.getAttr("lb", gp_vars)
        var_obj_coeffs: List[float] = self.model.getAttr("obj", gp_vars)
        constr_names: List[str] = self.model.getAttr("constrName", gp_constrs)
        varname_to_var: Dict = {}
        cname_to_constr: Dict = {}
        for (i, gp_var) in enumerate(gp_vars):
            assert var_names[i] not in varname_to_var, (
                f"Duplicated variable name detected: {var_names[i]}. "
                f"Unique variable names are currently required."
            )
            if var_types[i] == "I":
                assert var_ubs[i] == 1.0, (
                    "Only binary and continuous variables are currently supported. "
                    "Integer variable {var.varName} has upper bound {var.ub}."
                )
                assert var_lbs[i] == 0.0, (
                    "Only binary and continuous variables are currently supported. "
                    "Integer variable {var.varName} has lower bound {var.ub}."
                )
                var_types[i] = "B"
            assert var_types[i] in ["B", "C"], (
                "Only binary and continuous variables are currently supported. "
                "Variable {var.varName} has type {vtype}."
            )
            varname_to_var[var_names[i]] = gp_var
        for (i, gp_constr) in enumerate(gp_constrs):
            assert constr_names[i] not in cname_to_constr, (
                f"Duplicated constraint name detected: {constr_names[i]}. "
                f"Unique constraint names are currently required."
            )
            cname_to_constr[constr_names[i]] = gp_constr
        self._varname_to_var = varname_to_var
        self._cname_to_constr = cname_to_constr
        self._gp_vars = tuple(gp_vars)
        self._gp_constrs = tuple(gp_constrs)
        self._var_names = tuple(var_names)
        self._constr_names = tuple(constr_names)
        self._var_types = tuple(var_types)
        self._var_lbs = tuple(var_lbs)
        self._var_ubs = tuple(var_ubs)
        self._var_obj_coeffs = tuple(var_obj_coeffs)

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
        z = model.addVar(vtype=GRB.CONTINUOUS, name="z", ub=self.capacity)
        model.addConstr(
            gp.quicksum(x[i] * self.weights[i] for i in range(n)) == z,
            "eq_capacity",
        )
        model.setObjective(
            gp.quicksum(x[i] * self.prices[i] for i in range(n)), GRB.MAXIMIZE
        )
        return model

    @overrides
    def enforce_lazy_constraint(
        self,
        solver: InternalSolver,
        model: Any,
        violation: Hashable,
    ) -> None:
        x0 = model.getVarByName("x[0]")
        model.cbLazy(x0 <= 0)
