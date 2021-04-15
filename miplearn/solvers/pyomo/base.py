#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import re
import sys
from io import StringIO
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
import pyomo
from overrides import overrides
from pyomo import environ as pe
from pyomo.core import Var, Suffix, Objective
from pyomo.core.base import _GeneralVarData
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.expr.numeric_expr import SumExpression, MonomialTermExpression
from pyomo.opt import TerminationCondition
from pyomo.opt.base.solvers import SolverFactory

from miplearn.features import Variable, VariableFeatures
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
        self._has_lp_solution = False
        self._has_mip_solution = False
        self._obj: Dict[str, float] = {}

        for (key, value) in params.items():
            self._pyomo_solver.options[key] = value

    @overrides
    def add_constraint(
        self,
        constr: Any,
        name: str,
    ) -> None:
        assert self.model is not None
        if isinstance(constr, Constraint):
            assert constr.lhs is not None
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
        self._termination_condition = ""
        self._has_lp_solution = False
        self._has_mip_solution = False

    @overrides
    def are_callbacks_supported(self) -> bool:
        return False

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
    def fix(self, solution: Solution) -> None:
        for (varname, value) in solution.items():
            if value is None:
                continue
            var = self._varname_to_var[varname]
            var.fix(value)
            self._pyomo_solver.update_var(var)

    @overrides
    def get_constraints(self, with_static: bool = True) -> Dict[str, Constraint]:
        assert self.model is not None

        constraints = {}
        for constr in self.model.component_objects(pyomo.core.Constraint):
            if isinstance(constr, pe.ConstraintList):
                for idx in constr:
                    name = f"{constr.name}[{idx}]"
                    assert name not in constraints
                    constraints[name] = self._parse_pyomo_constraint(
                        constr[idx],
                        with_static=with_static,
                    )
            else:
                name = constr.name
                assert name not in constraints
                constraints[name] = self._parse_pyomo_constraint(
                    constr,
                    with_static=with_static,
                )

        return constraints

    @overrides
    def get_constraint_attrs(self) -> List[str]:
        return [
            "dual_value",
            "lazy",
            "lhs",
            "rhs",
            "sense",
            "slack",
        ]

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
    def get_variables_old(self, with_static: bool = True) -> Dict[str, Variable]:
        assert self.model is not None
        variables = {}
        for var in self.model.component_objects(pyomo.core.Var):
            for idx in var:
                varname = f"{var}[{idx}]"
                if idx is None:
                    varname = str(var)
                variables[varname] = self._parse_pyomo_variable(
                    var[idx],
                    with_static=with_static,
                )
        return variables

    @overrides
    def get_variables(
        self,
        with_static: bool = True,
        with_sa: bool = True,
    ) -> VariableFeatures:
        assert self.model is not None

        names: List[str] = []
        types: List[str] = []
        upper_bounds: List[float] = []
        lower_bounds: List[float] = []
        obj_coeffs: List[float] = []
        reduced_costs: List[float] = []
        values: List[float] = []

        for (i, var) in enumerate(self.model.component_objects(pyomo.core.Var)):
            for idx in var:
                v = var[idx]

                if with_static:
                    # Variable name
                    if idx is None:
                        names.append(str(var))
                    else:
                        names.append(f"{var}[{idx}]")

                    # Variable type
                    if v.domain == pyomo.core.Binary:
                        types.append("B")
                    elif v.domain in [
                        pyomo.core.Reals,
                        pyomo.core.NonNegativeReals,
                        pyomo.core.NonPositiveReals,
                        pyomo.core.NegativeReals,
                        pyomo.core.PositiveReals,
                    ]:
                        types.append("C")
                    else:
                        raise Exception(f"unknown variable domain: {v.domain}")

                    # Bounds
                    lb, ub = v.bounds
                    upper_bounds.append(float(ub))
                    lower_bounds.append(float(lb))

                    # Objective coefficient
                    if v.name in self._obj:
                        obj_coeffs.append(self._obj[v.name])
                    else:
                        obj_coeffs.append(0.0)

                # Reduced costs
                if self._has_lp_solution:
                    reduced_costs.append(self.model.rc[v])

                # Values
                if self._has_lp_solution or self._has_mip_solution:
                    values.append(v.value)

        names_t: Optional[Tuple[str, ...]] = None
        types_t: Optional[Tuple[str, ...]] = None
        upper_bounds_t: Optional[Tuple[float, ...]] = None
        lower_bounds_t: Optional[Tuple[float, ...]] = None
        obj_coeffs_t: Optional[Tuple[float, ...]] = None
        reduced_costs_t: Optional[Tuple[float, ...]] = None
        values_t: Optional[Tuple[float, ...]] = None

        if with_static:
            names_t = tuple(names)
            types_t = tuple(types)
            upper_bounds_t = tuple(upper_bounds)
            lower_bounds_t = tuple(lower_bounds)
            obj_coeffs_t = tuple(obj_coeffs)

        if self._has_lp_solution:
            reduced_costs_t = tuple(reduced_costs)

        if self._has_lp_solution or self._has_mip_solution:
            values_t = tuple(values)

        return VariableFeatures(
            names=names_t,
            types=types_t,
            upper_bounds=upper_bounds_t,
            lower_bounds=lower_bounds_t,
            obj_coeffs=obj_coeffs_t,
            reduced_costs=reduced_costs_t,
            values=values_t,
        )

    @overrides
    def get_variable_attrs(self) -> List[str]:
        return [
            # "basis_status",
            "lower_bound",
            "obj_coeff",
            "reduced_cost",
            # "sa_lb_down",
            # "sa_lb_up",
            # "sa_obj_down",
            # "sa_obj_up",
            # "sa_ub_down",
            # "sa_ub_up",
            "type",
            "upper_bound",
            "value",
            # new attributes
            "names",
            # "basis_status",
            "categories",
            "lower_bounds",
            "obj_coeffs",
            "reduced_costs",
            # "sa_lb_down",
            # "sa_lb_up",
            # "sa_obj_down",
            # "sa_obj_up",
            # "sa_ub_down",
            # "sa_ub_up",
            "types",
            "upper_bounds",
            "user_features",
            "values",
        ]

    @overrides
    def is_constraint_satisfied(self, constr: Constraint, tol: float = 1e-6) -> bool:
        lhs = 0.0
        assert constr.lhs is not None
        for (varname, coeff) in constr.lhs.items():
            var = self._varname_to_var[varname]
            lhs += var.value * coeff
        if constr.sense == "<":
            return lhs <= constr.rhs + tol
        elif constr.sense == ">":
            return lhs >= constr.rhs - tol
        else:
            return abs(constr.rhs - lhs) < abs(tol)

    @overrides
    def is_infeasible(self) -> bool:
        return self._termination_condition == TerminationCondition.infeasible

    @overrides
    def remove_constraint(self, name: str) -> None:
        assert self.model is not None
        constr = self._cname_to_constr[name]
        del self._cname_to_constr[name]
        self.model.del_component(constr)
        self._pyomo_solver.remove_constraint(constr)

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
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        self.model.rc = Suffix(direction=Suffix.IMPORT)
        self.model.slack = Suffix(direction=Suffix.IMPORT)
        self._pyomo_solver.set_instance(model)
        self._update_obj()
        self._update_vars()
        self._update_constrs()

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
        self._has_mip_solution = False
        self._has_lp_solution = False
        if not self.is_infeasible():
            self._has_mip_solution = True
            lb = results["Problem"][0]["Lower bound"]
            ub = results["Problem"][0]["Upper bound"]
        return MIPSolveStats(
            mip_lower_bound=lb,
            mip_upper_bound=ub,
            mip_wallclock_time=total_wallclock_time,
            mip_sense=self._obj_sense,
            mip_log=log,
            mip_nodes=node_count,
            mip_warm_start_value=ws_value,
        )

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
        self._termination_condition = results["Solver"][0]["Termination condition"]
        self._restore_integrality()
        opt_value = None
        self._has_lp_solution = False
        self._has_mip_solution = False
        if not self.is_infeasible():
            opt_value = results["Problem"][0]["Lower bound"]
            self._has_lp_solution = True
        return LPSolveStats(
            lp_value=opt_value,
            lp_log=streams[0].getvalue(),
            lp_wallclock_time=results["Solver"][0]["Wallclock time"],
        )

    @overrides
    def relax(self) -> None:
        for var in self._bin_vars:
            lb, ub = var.bounds
            var.setlb(lb)
            var.setub(ub)
            var.domain = pyomo.core.base.set_types.Reals
            self._pyomo_solver.update_var(var)

    def _clear_warm_start(self) -> None:
        for var in self._all_vars:
            if not var.fixed:
                var.value = None
        self._is_warm_start_available = False

    @staticmethod
    def _extract(
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

    def _extract_node_count(self, log: str) -> Optional[int]:
        value = self._extract(log, self._get_node_count_regexp())
        if value is None:
            return None
        return int(value)

    def _extract_warm_start_value(self, log: str) -> Optional[float]:
        value = self._extract(log, self._get_warm_start_regexp())
        if value is None:
            return None
        return float(value)

    def _get_node_count_regexp(self) -> Optional[str]:
        return None

    def _get_warm_start_regexp(self) -> Optional[str]:
        return None

    def _parse_pyomo_variable(
        self,
        pyomo_var: pyomo.core.Var,
        with_static: bool = True,
    ) -> Variable:
        assert self.model is not None
        variable = Variable()

        if with_static:
            # Variable type
            vtype: Optional[str] = None
            if pyomo_var.domain == pyomo.core.Binary:
                vtype = "B"
            elif pyomo_var.domain in [
                pyomo.core.Reals,
                pyomo.core.NonNegativeReals,
                pyomo.core.NonPositiveReals,
                pyomo.core.NegativeReals,
                pyomo.core.PositiveReals,
            ]:
                vtype = "C"
            if vtype is None:
                raise Exception(f"unknown variable domain: {pyomo_var.domain}")
            variable.type = vtype

            # Bounds
            lb, ub = pyomo_var.bounds
            variable.upper_bound = float(ub)
            variable.lower_bound = float(lb)

            # Objective coefficient
            obj_coeff = 0.0
            if pyomo_var.name in self._obj:
                obj_coeff = self._obj[pyomo_var.name]
            variable.obj_coeff = obj_coeff

        # Reduced costs
        if pyomo_var in self.model.rc:
            variable.reduced_cost = self.model.rc[pyomo_var]

        variable.value = pyomo_var.value
        return variable

    def _parse_pyomo_constraint(
        self,
        pyomo_constr: pyomo.core.Constraint,
        with_static: bool = True,
    ) -> Constraint:
        assert self.model is not None
        constr = Constraint()

        if with_static:
            # Extract RHS and sense
            has_ub = pyomo_constr.has_ub()
            has_lb = pyomo_constr.has_lb()
            assert (
                (not has_lb)
                or (not has_ub)
                or pyomo_constr.upper() == pyomo_constr.lower()
            ), "range constraints not supported"
            if not has_ub:
                constr.sense = ">"
                constr.rhs = pyomo_constr.lower()
            elif not has_lb:
                constr.sense = "<"
                constr.rhs = pyomo_constr.upper()
            else:
                constr.sense = "="
                constr.rhs = pyomo_constr.upper()

            # Extract LHS
            constr.lhs = self._parse_pyomo_expr(pyomo_constr.body)

        # Extract solution attributes
        if self._has_lp_solution:
            constr.dual_value = self.model.dual[pyomo_constr]

        if self._has_mip_solution or self._has_lp_solution:
            constr.slack = self.model.slack[pyomo_constr]

        # Build constraint
        return constr

    def _parse_pyomo_expr(self, expr: Any) -> Dict[str, float]:
        lhs = {}
        if isinstance(expr, SumExpression):
            for term in expr._args_:
                if isinstance(term, MonomialTermExpression):
                    lhs[term._args_[1].name] = float(term._args_[0])
                elif isinstance(term, _GeneralVarData):
                    lhs[term.name] = 1.0
                else:
                    raise Exception(f"Unknown term type: {term.__class__.__name__}")
        elif isinstance(expr, _GeneralVarData):
            lhs[expr.name] = 1.0
        else:
            raise Exception(f"Unknown expression type: {expr.__class__.__name__}")
        return lhs

    def _restore_integrality(self) -> None:
        for var in self._bin_vars:
            var.domain = pyomo.core.base.set_types.Binary
            self._pyomo_solver.update_var(var)

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
                varname = f"{var.name}[{idx}]"
                if idx is None:
                    varname = var.name
                self._varname_to_var[varname] = var[idx]
                self._all_vars += [var[idx]]
                if var[idx].domain == pyomo.core.base.set_types.Binary:
                    self._bin_vars += [var[idx]]
        for obj in self.model.component_objects(Objective):
            self._obj = self._parse_pyomo_expr(obj.expr)
            break

    def _update_constrs(self) -> None:
        assert self.model is not None
        self._cname_to_constr.clear()
        for constr in self.model.component_objects(pyomo.core.Constraint):
            if isinstance(constr, pe.ConstraintList):
                for idx in constr:
                    self._cname_to_constr[f"{constr.name}[{idx}]"] = constr[idx]
            else:
                self._cname_to_constr[constr.name] = constr


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
        model.z = pe.Var(domain=pe.Reals, bounds=(0, self.capacity))
        model.OBJ = pe.Objective(
            expr=sum(model.x[v] * self.prices[v] for v in items),
            sense=pe.maximize,
        )
        model.eq_capacity = pe.Constraint(
            expr=sum(model.x[v] * self.weights[v] for v in items) == model.z
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

    @overrides
    def get_variable_category(self, var_name: VariableName) -> Optional[Category]:
        if var_name.startswith("x"):
            return "default"
        return None
