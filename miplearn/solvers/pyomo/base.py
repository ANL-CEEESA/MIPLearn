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
from pyomo.core import Var, Suffix, Objective
from pyomo.core.base import _GeneralVarData
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.expr.numeric_expr import SumExpression, MonomialTermExpression
from pyomo.opt import TerminationCondition
from pyomo.opt.base.solvers import SolverFactory
from scipy.sparse import coo_matrix

from miplearn.instance.base import Instance
from miplearn.solvers import _RedirectOutput, _none_if_empty
from miplearn.solvers.internal import (
    InternalSolver,
    LPSolveStats,
    IterationCallback,
    LazyCallback,
    MIPSolveStats,
    Variables,
    Constraints,
)
from miplearn.types import (
    SolverParams,
    UserCutCallback,
    Solution,
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
        self._varname_to_var: Dict[bytes, pe.Var] = {}
        self._varname_to_idx: Dict[str, int] = {}
        self._cname_to_constr: Dict[str, pe.Constraint] = {}
        self._termination_condition: str = ""
        self._has_lp_solution = False
        self._has_mip_solution = False
        self._obj: Dict[str, float] = {}

        for (key, value) in params.items():
            self._pyomo_solver.options[key] = value

    def add_constraint(
        self,
        constr: Any,
    ) -> None:
        assert self.model is not None
        self._pyomo_solver.add_constraint(constr)
        self._termination_condition = ""
        self._has_lp_solution = False
        self._has_mip_solution = False

    @overrides
    def add_constraints(self, cf: Constraints) -> None:
        assert cf.names is not None
        assert cf.senses is not None
        assert cf.lhs is not None
        assert cf.rhs is not None
        assert self.model is not None
        lhs = cf.lhs.tocsr()
        for i in range(len(cf.names)):
            row = lhs[i, :]
            lhsi = 0.0
            for j in range(row.getnnz()):
                lhsi += self._all_vars[row.indices[j]] * row.data[j]
            if cf.senses[i] == b"=":
                expr = lhsi == cf.rhs[i]
            elif cf.senses[i] == b"<":
                expr = lhsi <= cf.rhs[i]
            elif cf.senses[i] == b">":
                expr = lhsi >= cf.rhs[i]
            else:
                raise Exception(f"Unknown sense: {cf.senses[i]}")
            cl = pe.Constraint(expr=expr, name=cf.names[i])
            self.model.add_component(cf.names[i].decode(), cl)
            self._pyomo_solver.add_constraint(cl)
            self._cname_to_constr[cf.names[i]] = cl
        self._termination_condition = ""
        self._has_lp_solution = False
        self._has_mip_solution = False

    @overrides
    def are_callbacks_supported(self) -> bool:
        return False

    @overrides
    def are_constraints_satisfied(
        self,
        cf: Constraints,
        tol: float = 1e-5,
    ) -> List[bool]:
        assert cf.names is not None
        assert cf.lhs is not None
        assert cf.rhs is not None
        assert cf.senses is not None
        x = [v.value for v in self._all_vars]
        lhs = cf.lhs.tocsr() * x
        result = []
        for i in range(len(lhs)):
            if cf.senses[i] == b"<":
                result.append(lhs[i] <= cf.rhs[i] + tol)
            elif cf.senses[i] == b">":
                result.append(lhs[i] >= cf.rhs[i] - tol)
            elif cf.senses[i] == b"=":
                result.append(abs(cf.rhs[i] - lhs[i]) < tol)
            else:
                raise Exception(f"unknown sense: {cf.senses[i]}")
        return result

    @overrides
    def build_test_instance_infeasible(self) -> Instance:
        return PyomoTestInstanceInfeasible()

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
    def get_constraints(
        self,
        with_static: bool = True,
        with_sa: bool = True,
        with_lhs: bool = True,
    ) -> Constraints:
        model = self.model
        assert model is not None
        names: List[str] = []
        rhs: List[float] = []
        senses: List[str] = []
        dual_values: List[float] = []
        slacks: List[float] = []
        lhs_row: List[int] = []
        lhs_col: List[int] = []
        lhs_data: List[float] = []
        lhs: Optional[coo_matrix] = None

        def _parse_constraint(c: pe.Constraint, row: int) -> None:
            assert model is not None
            if with_static:
                # Extract RHS and sense
                has_ub = c.has_ub()
                has_lb = c.has_lb()
                assert (
                    (not has_lb) or (not has_ub) or c.upper() == c.lower()
                ), "range constraints not supported"
                if not has_ub:
                    senses.append(">")
                    rhs.append(float(c.lower()))
                elif not has_lb:
                    senses.append("<")
                    rhs.append(float(c.upper()))
                else:
                    senses.append("=")
                    rhs.append(float(c.upper()))

                if with_lhs:
                    # Extract LHS
                    expr = c.body
                    if isinstance(expr, SumExpression):
                        for term in expr._args_:
                            if isinstance(term, MonomialTermExpression):
                                lhs_row.append(row)
                                lhs_col.append(
                                    self._varname_to_idx[term._args_[1].name]
                                )
                                lhs_data.append(float(term._args_[0]))
                            elif isinstance(term, _GeneralVarData):
                                lhs_row.append(row)
                                lhs_col.append(self._varname_to_idx[term.name])
                                lhs_data.append(1.0)
                            else:
                                raise Exception(
                                    f"Unknown term type: {term.__class__.__name__}"
                                )
                    elif isinstance(expr, _GeneralVarData):
                        lhs_row.append(row)
                        lhs_col.append(self._varname_to_idx[expr.name])
                        lhs_data.append(1.0)
                    else:
                        raise Exception(
                            f"Unknown expression type: {expr.__class__.__name__}"
                        )

            # Extract dual values
            if self._has_lp_solution:
                dual_values.append(model.dual[c])

            # Extract slacks
            if self._has_mip_solution or self._has_lp_solution:
                slacks.append(model.slack[c])

        curr_row = 0
        for (i, constr) in enumerate(model.component_objects(pyomo.core.Constraint)):
            if isinstance(constr, pe.ConstraintList):
                for idx in constr:
                    names.append(constr[idx].name)
                    _parse_constraint(constr[idx], curr_row)
                    curr_row += 1
            else:
                names.append(constr.name)
                _parse_constraint(constr, curr_row)
                curr_row += 1

        if len(lhs_data) > 0:
            lhs = coo_matrix((lhs_data, (lhs_row, lhs_col))).tocoo()

        return Constraints(
            names=_none_if_empty(np.array(names, dtype="S")),
            rhs=_none_if_empty(np.array(rhs, dtype=float)),
            senses=_none_if_empty(np.array(senses, dtype="S")),
            lhs=lhs,
            slacks=_none_if_empty(np.array(slacks, dtype=float)),
            dual_values=_none_if_empty(np.array(dual_values, dtype=float)),
        )

    @overrides
    def get_constraint_attrs(self) -> List[str]:
        return [
            "dual_values",
            "lhs",
            "names",
            "rhs",
            "senses",
            "slacks",
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
                solution[var[index].name.encode()] = var[index].value
        return solution

    @overrides
    def get_variables(
        self,
        with_static: bool = True,
        with_sa: bool = True,
    ) -> Variables:
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

                # Variable name
                if idx is None:
                    names.append(var.name)
                else:
                    names.append(var[idx].name)

                if with_static:
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

        return Variables(
            names=_none_if_empty(np.array(names, dtype="S")),
            types=_none_if_empty(np.array(types, dtype="S")),
            upper_bounds=_none_if_empty(np.array(upper_bounds, dtype=float)),
            lower_bounds=_none_if_empty(np.array(lower_bounds, dtype=float)),
            obj_coeffs=_none_if_empty(np.array(obj_coeffs, dtype=float)),
            reduced_costs=_none_if_empty(np.array(reduced_costs, dtype=float)),
            values=_none_if_empty(np.array(values, dtype=float)),
        )

    @overrides
    def get_variable_attrs(self) -> List[str]:
        return [
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
    def is_infeasible(self) -> bool:
        return self._termination_condition == TerminationCondition.infeasible

    @overrides
    def remove_constraints(self, names: List[str]) -> None:
        assert self.model is not None
        for name in names:
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
        self._relax()
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

    def _relax(self) -> None:
        for var in self._bin_vars:
            lb, ub = var.bounds
            var.setlb(lb)
            var.setub(ub)
            var.domain = pyomo.core.base.set_types.Reals
            self._pyomo_solver.update_var(var)

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
        self._varname_to_idx = {}
        for var in self.model.component_objects(Var):
            for idx in var:
                varname = var.name
                if idx is not None:
                    varname = var[idx].name
                self._varname_to_var[varname.encode()] = var[idx]
                self._varname_to_idx[varname] = len(self._all_vars)
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
                    self._cname_to_constr[constr[idx].name] = constr[idx]
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
        self.n = len(weights)

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
    def get_instance_features(self) -> np.ndarray:
        return np.array(
            [
                self.capacity,
                np.average(self.weights),
            ]
        )

    @overrides
    def get_variable_features(self, names: np.ndarray) -> np.ndarray:
        return np.vstack(
            [
                [[self.weights[i], self.prices[i]] for i in range(self.n)],
                [0.0, 0.0],
            ]
        )

    @overrides
    def get_variable_categories(self, names: np.ndarray) -> np.ndarray:
        return np.array(
            ["default" if n.decode().startswith("x") else "" for n in names],
            dtype="S",
        )
