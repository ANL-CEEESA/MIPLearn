#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import re
import sys
from io import StringIO
from typing import Any, List, Dict, Optional

import numpy as np
import pyomo
from overrides import overrides
from pyomo import environ as pe
from pyomo.core import Var
from pyomo.core.base import _GeneralVarData
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.expr.numeric_expr import SumExpression, MonomialTermExpression
from pyomo.opt import TerminationCondition
from pyomo.opt.base.solvers import SolverFactory

from miplearn.instance.base import Instance
from miplearn.solvers import _RedirectOutput
from miplearn.solvers.internal import (
    InternalSolver,
    LPSolveStats,
    IterationCallback,
    LazyCallback,
    MIPSolveStats,
    Constraint,
)
from miplearn.types import (
    SolverParams,
    UserCutCallback,
    Solution,
    VariableName,
    Category,
)

logger = logging.getLogger(__name__)


class BasePyomoSolver(InternalSolver):
    """
    Base class for all Pyomo solvers.
    """

    def __init__(
        self,
        solver_factory: SolverFactory,
        params: SolverParams,
    ) -> None:
        self.instance: Optional[Instance] = None
        self.model: Optional[pe.ConcreteModel] = None
        self.params = params
        self._all_vars: List[pe.Var] = []
        self._bin_vars: List[pe.Var] = []
        self._is_warm_start_available: bool = False
        self._pyomo_solver: SolverFactory = solver_factory
        self._obj_sense: str = "min"
        self._varname_to_var: Dict[str, pe.Var] = {}
        self._cname_to_constr: Dict[str, pe.Constraint] = {}
        self._termination_condition: str = ""

        for (key, value) in params.items():
            self._pyomo_solver.options[key] = value

    @overrides
    def solve_lp(
        self,
        tee: bool = False,
    ) -> LPSolveStats:
        self.relax()
        streams: List[Any] = [StringIO()]
        if tee:
            streams += [sys.stdout]
        with _RedirectOutput(streams):
            results = self._pyomo_solver.solve(tee=True)
        self._restore_integrality()
        opt_value = None
        if not self.is_infeasible():
            opt_value = results["Problem"][0]["Lower bound"]
        return {
            "LP value": opt_value,
            "LP log": streams[0].getvalue(),
        }

    def _restore_integrality(self) -> None:
        for var in self._bin_vars:
            var.domain = pyomo.core.base.set_types.Binary
            self._pyomo_solver.update_var(var)

    @overrides
    def solve(
        self,
        tee: bool = False,
        iteration_cb: Optional[IterationCallback] = None,
        lazy_cb: Optional[LazyCallback] = None,
        user_cut_cb: Optional[UserCutCallback] = None,
    ) -> MIPSolveStats:
        assert lazy_cb is None, "callbacks are not currently supported"
        assert user_cut_cb is None, "callbacks are not currently supported"
        total_wallclock_time = 0
        streams: List[Any] = [StringIO()]
        if tee:
            streams += [sys.stdout]
        if iteration_cb is None:
            iteration_cb = lambda: False
        while True:
            logger.debug("Solving MIP...")
            with _RedirectOutput(streams):
                results = self._pyomo_solver.solve(
                    tee=True,
                    warmstart=self._is_warm_start_available,
                )
            total_wallclock_time += results["Solver"][0]["Wallclock time"]
            should_repeat = iteration_cb()
            if not should_repeat:
                break
        log = streams[0].getvalue()
        node_count = self._extract_node_count(log)
        ws_value = self._extract_warm_start_value(log)
        self._termination_condition = results["Solver"][0]["Termination condition"]
        lb, ub = None, None
        if not self.is_infeasible():
            lb = results["Problem"][0]["Lower bound"]
            ub = results["Problem"][0]["Upper bound"]
        stats: MIPSolveStats = {
            "Lower bound": lb,
            "Upper bound": ub,
            "Wallclock time": total_wallclock_time,
            "Sense": self._obj_sense,
            "MIP log": log,
            "Nodes": node_count,
            "Warm start value": ws_value,
        }
        return stats

    @overrides
    def get_solution(self) -> Optional[Solution]:
        assert self.model is not None
        if self.is_infeasible():
            return None
        solution: Solution = {}
        for var in self.model.component_objects(Var):
            for index in var:
                if var[index].fixed:
                    continue
                solution[f"{var}[{index}]"] = var[index].value
        return solution

    @overrides
    def get_variable_names(self) -> List[VariableName]:
        assert self.model is not None
        variables: List[VariableName] = []
        for var in self.model.component_objects(Var):
            for index in var:
                if var[index].fixed:
                    continue
                variables += [f"{var}[{index}]"]
        return variables

    @overrides
    def set_warm_start(self, solution: Solution) -> None:
        self._clear_warm_start()
        count_fixed = 0
        for (var_name, value) in solution.items():
            if value is None:
                continue
            var = self._varname_to_var[var_name]
            var.value = solution[var_name]
            count_fixed += 1
        if count_fixed > 0:
            self._is_warm_start_available = True

    @overrides
    def set_instance(
        self,
        instance: Instance,
        model: Any = None,
    ) -> None:
        if model is None:
            model = instance.to_model()
        assert isinstance(model, pe.ConcreteModel)
        self.instance = instance
        self.model = model
        self.model.extra_constraints = ConstraintList()
        self._pyomo_solver.set_instance(model)
        self._update_obj()
        self._update_vars()
        self._update_constrs()

    def _clear_warm_start(self) -> None:
        for var in self._all_vars:
            if not var.fixed:
                var.value = None
        self._is_warm_start_available = False

    def _update_obj(self) -> None:
        self._obj_sense = "max"
        if self._pyomo_solver._objective.sense == pyomo.core.kernel.objective.minimize:
            self._obj_sense = "min"

    def _update_vars(self) -> None:
        assert self.model is not None
        self._all_vars = []
        self._bin_vars = []
        self._varname_to_var = {}
        for var in self.model.component_objects(Var):
            for idx in var:
                self._varname_to_var[f"{var.name}[{idx}]"] = var[idx]
                self._all_vars += [var[idx]]
                if var[idx].domain == pyomo.core.base.set_types.Binary:
                    self._bin_vars += [var[idx]]

    def _update_constrs(self) -> None:
        assert self.model is not None
        self._cname_to_constr.clear()
        for constr in self.model.component_objects(pyomo.core.Constraint):
            if isinstance(constr, pe.ConstraintList):
                for idx in constr:
                    self._cname_to_constr[f"{constr.name}[{idx}]"] = constr[idx]
            else:
                self._cname_to_constr[constr.name] = constr

    @overrides
    def fix(self, solution: Solution) -> None:
        for (varname, value) in solution.items():
            if value is None:
                continue
            var = self._varname_to_var[varname]
            var.fix(value)
            self._pyomo_solver.update_var(var)

    @overrides
    def add_constraint(
        self,
        constr: Any,
        name: str,
    ) -> None:
        assert self.model is not None
        if isinstance(constr, Constraint):
            lhs = 0.0
            for (varname, coeff) in constr.lhs.items():
                var = self._varname_to_var[varname]
                lhs += var * coeff
            if constr.sense == "=":
                expr = lhs == constr.rhs
            elif constr.sense == "<":
                expr = lhs <= constr.rhs
            else:
                expr = lhs >= constr.rhs
            cl = pe.Constraint(expr=expr, name=name)
            self.model.add_component(name, cl)
            self._pyomo_solver.add_constraint(cl)
            self._cname_to_constr[name] = cl
        else:
            self._pyomo_solver.add_constraint(constr)

    @overrides
    def remove_constraint(self, name: str) -> None:
        assert self.model is not None
        constr = self._cname_to_constr[name]
        del self._cname_to_constr[name]
        self.model.del_component(constr)
        self._pyomo_solver.remove_constraint(constr)

    @overrides
    def is_constraint_satisfied(self, constr: Constraint, tol: float = 1e-6) -> bool:
        lhs = 0.0
        for (varname, coeff) in constr.lhs.items():
            var = self._varname_to_var[varname]
            lhs += var.value * coeff
        if constr.sense == "<":
            return lhs <= constr.rhs + tol
        elif constr.sense == ">":
            return lhs >= constr.rhs - tol
        else:
            return abs(constr.rhs - lhs) < abs(tol)

    @staticmethod
    def __extract(
        log: str,
        regexp: Optional[str],
        default: Optional[str] = None,
    ) -> Optional[str]:
        if regexp is None:
            return default
        value = default
        for line in log.splitlines():
            matches = re.findall(regexp, line)
            if len(matches) == 0:
                continue
            value = matches[0]
        return value

    def _extract_warm_start_value(self, log: str) -> Optional[float]:
        value = self.__extract(log, self._get_warm_start_regexp())
        if value is None:
            return None
        return float(value)

    def _extract_node_count(self, log: str) -> Optional[int]:
        value = self.__extract(log, self._get_node_count_regexp())
        if value is None:
            return None
        return int(value)

    def _get_warm_start_regexp(self) -> Optional[str]:
        return None

    def _get_node_count_regexp(self) -> Optional[str]:
        return None

    @overrides
    def relax(self) -> None:
        for var in self._bin_vars:
            lb, ub = var.bounds
            var.setlb(lb)
            var.setub(ub)
            var.domain = pyomo.core.base.set_types.Reals
            self._pyomo_solver.update_var(var)

    @overrides
    def get_inequality_slacks(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for (cname, cobj) in self._cname_to_constr.items():
            if cobj.equality:
                continue
            result[cname] = cobj.slack()
        return result

    @overrides
    def is_infeasible(self) -> bool:
        return self._termination_condition == TerminationCondition.infeasible

    @overrides
    def get_dual(self, cid: str) -> float:
        raise NotImplementedError()

    @overrides
    def get_sense(self) -> str:
        return self._obj_sense

    @overrides
    def build_test_instance_infeasible(self) -> Instance:
        return PyomoTestInstanceInfeasible()

    @overrides
    def build_test_instance_redundancy(self) -> Instance:
        return PyomoTestInstanceRedundancy()

    @overrides
    def build_test_instance_knapsack(self) -> Instance:
        return PyomoTestInstanceKnapsack(
            weights=[23.0, 26.0, 20.0, 18.0],
            prices=[505.0, 352.0, 458.0, 220.0],
            capacity=67.0,
        )

    @overrides
    def get_constraints(self) -> Dict[str, Constraint]:
        assert self.model is not None

        constraints = {}
        for constr in self.model.component_objects(pyomo.core.Constraint):
            if isinstance(constr, pe.ConstraintList):
                for idx in constr:
                    name = f"{constr.name}[{idx}]"
                    assert name not in constraints
                    constraints[name] = self._parse_pyomo_constraint(constr[idx])
            else:
                name = constr.name
                assert name not in constraints
                constraints[name] = self._parse_pyomo_constraint(constr)

        return constraints

    @staticmethod
    def _parse_pyomo_constraint(c: pyomo.core.Constraint) -> Constraint:
        # Extract RHS and sense
        has_ub = c.has_ub()
        has_lb = c.has_lb()
        assert (
            (not has_lb) or (not has_ub) or c.upper() == c.lower()
        ), "range constraints not supported"
        if has_lb:
            sense = ">"
            rhs = c.lower()
        elif has_ub:
            sense = "<"
            rhs = c.upper()
        else:
            sense = "="
            rhs = c.upper()

        # Extract LHS
        lhs = {}
        if isinstance(c.body, SumExpression):
            for term in c.body._args_:
                if isinstance(term, MonomialTermExpression):
                    lhs[term._args_[1].name] = term._args_[0]
                elif isinstance(term, _GeneralVarData):
                    lhs[term.name] = 1.0
                else:
                    raise Exception(f"Unknown term type: {term.__class__.__name__}")
        elif isinstance(c.body, _GeneralVarData):
            lhs[c.body.name] = 1.0
        else:
            raise Exception(f"Unknown expression type: {c.body.__class__.__name__}")

        # Build constraint
        return Constraint(
            lhs=lhs,
            rhs=rhs,
            sense=sense,
        )

    @overrides
    def are_callbacks_supported(self) -> bool:
        return False


class PyomoTestInstanceInfeasible(Instance):
    @overrides
    def to_model(self) -> pe.ConcreteModel:
        model = pe.ConcreteModel()
        model.x = pe.Var([0], domain=pe.Binary)
        model.OBJ = pe.Objective(expr=model.x[0], sense=pe.maximize)
        model.eq = pe.Constraint(expr=model.x[0] >= 2)
        return model


class PyomoTestInstanceRedundancy(Instance):
    @overrides
    def to_model(self) -> pe.ConcreteModel:
        model = pe.ConcreteModel()
        model.x = pe.Var([0, 1], domain=pe.Binary)
        model.OBJ = pe.Objective(expr=model.x[0] + model.x[1], sense=pe.maximize)
        model.eq1 = pe.Constraint(expr=model.x[0] + model.x[1] <= 1)
        model.eq2 = pe.Constraint(expr=model.x[0] + model.x[1] <= 2)
        return model


class PyomoTestInstanceKnapsack(Instance):
    """
    Simpler (one-dimensional) Knapsack Problem, used for testing.
    """

    def __init__(
        self,
        weights: List[float],
        prices: List[float],
        capacity: float,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.prices = prices
        self.capacity = capacity
        self.varname_to_item: Dict[VariableName, int] = {
            f"x[{i}]": i for i in range(len(self.weights))
        }

    @overrides
    def to_model(self) -> pe.ConcreteModel:
        model = pe.ConcreteModel()
        items = range(len(self.weights))
        model.x = pe.Var(items, domain=pe.Binary)
        model.OBJ = pe.Objective(
            expr=sum(model.x[v] * self.prices[v] for v in items),
            sense=pe.maximize,
        )
        model.eq_capacity = pe.Constraint(
            expr=sum(model.x[v] * self.weights[v] for v in items) <= self.capacity
        )
        return model

    @overrides
    def get_instance_features(self) -> List[float]:
        return [
            self.capacity,
            np.average(self.weights),
        ]

    @overrides
    def get_variable_features(self, var_name: VariableName) -> List[Category]:
        item = self.varname_to_item[var_name]
        return [
            self.weights[item],
            self.prices[item],
        ]
