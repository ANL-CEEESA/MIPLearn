#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from numbers import Number
from typing import Optional, Dict, List, Any, Tuple, Callable

import numpy as np
import pyomo
import pyomo.environ as pe
from pyomo.core import Objective, Var, Suffix
from pyomo.core.base import _GeneralVarData
from pyomo.core.expr.numeric_expr import SumExpression, MonomialTermExpression
from scipy.sparse import coo_matrix

from miplearn.h5 import H5File
from miplearn.solvers.abstract import AbstractModel
from miplearn.solvers.gurobi import _gurobi_callback, _gurobi_add_constr


class PyomoModel(AbstractModel):
    def __init__(
        self,
        model: pe.ConcreteModel,
        solver_name: str = "gurobi_persistent",
        lazy_separate: Optional[Callable] = None,
        lazy_enforce: Optional[Callable] = None,
    ):
        super().__init__()
        self.inner = model
        self.solver_name = solver_name
        self.solver = pe.SolverFactory(solver_name)
        self.is_persistent = hasattr(self.solver, "set_instance")
        if self.is_persistent:
            self.solver.set_instance(model)
        self.results: Optional[Dict] = None
        self._is_warm_start_available = False
        self.lazy_separate = lazy_separate
        self.lazy_enforce = lazy_enforce
        self.lazy_constrs_: Optional[List[Any]] = None
        if not hasattr(self.inner, "dual"):
            self.inner.dual = Suffix(direction=Suffix.IMPORT)
            self.inner.rc = Suffix(direction=Suffix.IMPORT)
            self.inner.slack = Suffix(direction=Suffix.IMPORT)

    def add_constr(self, constr: Any) -> None:
        assert (
            self.solver_name == "gurobi_persistent"
        ), "Callbacks are currently only supported on gurobi_persistent"
        _gurobi_add_constr(self.solver, self.where, constr)

    def add_constrs(
        self,
        var_names: np.ndarray,
        constrs_lhs: np.ndarray,
        constrs_sense: np.ndarray,
        constrs_rhs: np.ndarray,
        stats: Optional[Dict] = None,
    ) -> None:
        variables = self._var_names_to_vars(var_names)
        if not hasattr(self.inner, "added_eqs"):
            self.inner.added_eqs = pe.ConstraintList()
        for i in range(len(constrs_sense)):
            lhs = sum([variables[j] * constrs_lhs[i, j] for j in range(len(variables))])
            sense = constrs_sense[i]
            rhs = constrs_rhs[i]
            if sense == b"=":
                eq = self.inner.added_eqs.add(lhs == rhs)
            elif sense == b"<":
                eq = self.inner.added_eqs.add(lhs <= rhs)
            elif sense == b">":
                eq = self.inner.added_eqs.add(lhs >= rhs)
            else:
                raise Exception(f"Unknown sense: {sense}")
            self.solver.add_constraint(eq)

    def _var_names_to_vars(self, var_names: np.ndarray) -> List[Any]:
        varname_to_var = {}
        for var in self.inner.component_objects(Var):
            for idx in var:
                v = var[idx]
                varname_to_var[v.name] = var[idx]
        return [varname_to_var[var_name.decode()] for var_name in var_names]

    def extract_after_load(self, h5: H5File) -> None:
        self._extract_after_load_vars(h5)
        self._extract_after_load_constrs(h5)
        h5.put_scalar("static_sense", self._get_sense())

    def extract_after_lp(self, h5: H5File) -> None:
        assert self.results is not None
        self._extract_after_lp_vars(h5)
        self._extract_after_lp_constrs(h5)
        h5.put_scalar("lp_obj_value", self.results["Problem"][0]["Lower bound"])
        h5.put_scalar("lp_wallclock_time", self._get_runtime())

    def _get_runtime(self) -> float:
        assert self.results is not None
        solver_dict = self.results["Solver"][0]
        for key in ["Wallclock time", "User time"]:
            if isinstance(solver_dict[key], Number):
                return solver_dict[key]
        raise Exception("Time unavailable")

    def extract_after_mip(self, h5: H5File) -> None:
        assert self.results is not None
        h5.put_scalar("mip_wallclock_time", self._get_runtime())
        if self.results["Solver"][0]["Termination condition"] == "infeasible":
            return
        self._extract_after_mip_vars(h5)
        self._extract_after_mip_constrs(h5)
        if self._get_sense() == "max":
            obj_value = self.results["Problem"][0]["Lower bound"]
            obj_bound = self.results["Problem"][0]["Upper bound"]
        else:
            obj_value = self.results["Problem"][0]["Upper bound"]
            obj_bound = self.results["Problem"][0]["Lower bound"]
        h5.put_scalar("mip_obj_value", obj_value)
        h5.put_scalar("mip_obj_bound", obj_bound)
        h5.put_scalar("mip_gap", self._gap(obj_value, obj_bound))

    def fix_variables(
        self,
        var_names: np.ndarray,
        var_values: np.ndarray,
        stats: Optional[Dict] = None,
    ) -> None:
        variables = self._var_names_to_vars(var_names)
        for (var, val) in zip(variables, var_values):
            if np.isfinite(val):
                var.fix(val)
                self.solver.update_var(var)

    def optimize(self) -> None:
        self.lazy_constrs_ = []

        if self.lazy_separate is not None:
            assert (
                self.solver_name == "gurobi_persistent"
            ), "Callbacks are currently only supported on gurobi_persistent"

            def callback(_: Any, __: Any, where: int) -> None:
                _gurobi_callback(self, where)

            self.solver.set_gurobi_param("PreCrush", 1)
            self.solver.set_gurobi_param("LazyConstraints", 1)
            self.solver.set_callback(callback)

        if self.is_persistent:
            self.results = self.solver.solve(
                tee=True,
                warmstart=self._is_warm_start_available,
            )
        else:
            self.results = self.solver.solve(
                self.inner,
                tee=True,
            )

    def relax(self) -> "AbstractModel":
        relaxed = self.inner.clone()
        for var in relaxed.component_objects(Var):
            for idx in var:
                if var[idx].domain == pyomo.core.base.set_types.Binary:
                    lb, ub = var[idx].bounds
                    var[idx].setlb(lb)
                    var[idx].setub(ub)
                    var[idx].domain = pyomo.core.base.set_types.Reals
        return PyomoModel(relaxed, self.solver_name)

    def set_warm_starts(
        self,
        var_names: np.ndarray,
        var_values: np.ndarray,
        stats: Optional[Dict] = None,
    ) -> None:
        assert len(var_values.shape) == 2
        (n_starts, n_vars) = var_values.shape
        assert len(var_names.shape) == 1
        assert var_names.shape[0] == n_vars
        assert n_starts == 1, "Pyomo does not support multiple warm starts"
        variables = self._var_names_to_vars(var_names)
        for (var, val) in zip(variables, var_values[0, :]):
            if np.isfinite(val):
                var.value = val
        self._is_warm_start_available = True

    def _extract_after_load_vars(self, h5: H5File) -> None:
        names: List[str] = []
        types: List[str] = []
        upper_bounds: List[float] = []
        lower_bounds: List[float] = []
        obj_coeffs: List[float] = []

        obj = None
        obj_offset = 0.0
        obj_count = 0
        for obj in self.inner.component_objects(Objective):
            obj, obj_offset = self._parse_pyomo_expr(obj.expr)
            obj_count += 1
        assert obj_count == 1, f"One objective function expected; found {obj_count}"

        for (i, var) in enumerate(self.inner.component_objects(pyomo.core.Var)):
            for idx in var:
                v = var[idx]

                # Variable name
                if idx is None:
                    names.append(var.name)
                else:
                    names.append(var[idx].name)

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

                # Variable upper/lower bounds
                lb, ub = v.bounds
                if lb is None:
                    lb = -float("inf")
                if ub is None:
                    ub = float("Inf")
                upper_bounds.append(float(ub))
                lower_bounds.append(float(lb))

                # Objective coefficients
                if v.name in obj:
                    obj_coeffs.append(obj[v.name])
                else:
                    obj_coeffs.append(0.0)

        h5.put_array("static_var_names", np.array(names, dtype="S"))
        h5.put_array("static_var_types", np.array(types, dtype="S"))
        h5.put_array("static_var_lower_bounds", np.array(lower_bounds))
        h5.put_array("static_var_upper_bounds", np.array(upper_bounds))
        h5.put_array("static_var_obj_coeffs", np.array(obj_coeffs))
        h5.put_scalar("static_obj_offset", obj_offset)

    def _extract_after_load_constrs(self, h5: H5File) -> None:
        names: List[str] = []
        rhs: List[float] = []
        senses: List[str] = []
        lhs_row: List[int] = []
        lhs_col: List[int] = []
        lhs_data: List[float] = []

        varname_to_idx: Dict[str, int] = {}
        for var in self.inner.component_objects(Var):
            for idx in var:
                varname = var.name
                if idx is not None:
                    varname = var[idx].name
                varname_to_idx[varname] = len(varname_to_idx)

        def _parse_constraint(c: pe.Constraint, row: int) -> None:
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

            # Extract LHS
            expr = c.body
            if isinstance(expr, SumExpression):
                for term in expr._args_:
                    if isinstance(term, MonomialTermExpression):
                        lhs_row.append(row)
                        lhs_col.append(varname_to_idx[term._args_[1].name])
                        lhs_data.append(float(term._args_[0]))
                    elif isinstance(term, _GeneralVarData):
                        lhs_row.append(row)
                        lhs_col.append(varname_to_idx[term.name])
                        lhs_data.append(1.0)
                    else:
                        raise Exception(f"Unknown term type: {term.__class__.__name__}")
            elif isinstance(expr, _GeneralVarData):
                lhs_row.append(row)
                lhs_col.append(varname_to_idx[expr.name])
                lhs_data.append(1.0)
            else:
                raise Exception(f"Unknown expression type: {expr.__class__.__name__}")

        curr_row = 0
        for (i, constr) in enumerate(
            self.inner.component_objects(pyomo.core.Constraint)
        ):
            if len(constr) > 0:
                for idx in constr:
                    names.append(constr[idx].name)
                    _parse_constraint(constr[idx], curr_row)
                    curr_row += 1
            else:
                names.append(constr.name)
                _parse_constraint(constr, curr_row)
                curr_row += 1

        lhs = coo_matrix((lhs_data, (lhs_row, lhs_col))).tocoo()
        h5.put_sparse("static_constr_lhs", lhs)
        h5.put_array("static_constr_names", np.array(names, dtype="S"))
        h5.put_array("static_constr_rhs", np.array(rhs))
        h5.put_array("static_constr_sense", np.array(senses, dtype="S"))

    def _extract_after_lp_vars(self, h5: H5File) -> None:
        rc = []
        values = []
        for var in self.inner.component_objects(Var):
            for idx in var:
                v = var[idx]
                rc.append(self.inner.rc[v])
                values.append(v.value)
        h5.put_array("lp_var_reduced_costs", np.array(rc))
        h5.put_array("lp_var_values", np.array(values))

    def _extract_after_lp_constrs(self, h5: H5File) -> None:
        dual = []
        slacks = []
        for constr in self.inner.component_objects(pyomo.core.Constraint):
            for idx in constr:
                c = constr[idx]
                dual.append(self.inner.dual[c])
                slacks.append(abs(self.inner.slack[c]))
        h5.put_array("lp_constr_dual_values", np.array(dual))
        h5.put_array("lp_constr_slacks", np.array(slacks))

    def _extract_after_mip_vars(self, h5: H5File) -> None:
        values = []
        for var in self.inner.component_objects(Var):
            for idx in var:
                v = var[idx]
                values.append(v.value)
        h5.put_array("mip_var_values", np.array(values))

    def _extract_after_mip_constrs(self, h5: H5File) -> None:
        slacks = []
        for constr in self.inner.component_objects(pyomo.core.Constraint):
            for idx in constr:
                c = constr[idx]
                slacks.append(abs(self.inner.slack[c]))
        h5.put_array("mip_constr_slacks", np.array(slacks))

    def _parse_pyomo_expr(self, expr: Any) -> Tuple[Dict[str, float], float]:
        lhs = {}
        offset = 0.0
        if isinstance(expr, SumExpression):
            for term in expr._args_:
                if isinstance(term, MonomialTermExpression):
                    lhs[term._args_[1].name] = float(term._args_[0])
                elif isinstance(term, _GeneralVarData):
                    lhs[term.name] = 1.0
                elif isinstance(term, float):
                    offset += term
                else:
                    raise Exception(f"Unknown term type: {term.__class__.__name__}")
        elif isinstance(expr, _GeneralVarData):
            lhs[expr.name] = 1.0
        else:
            raise Exception(f"Unknown expression type: {expr.__class__.__name__}")
        return lhs, offset

    def _gap(self, zp: float, zd: float, tol: float = 1e-6) -> float:
        # Reference: https://www.gurobi.com/documentation/9.5/refman/mipgap2.html
        if abs(zp) < tol:
            if abs(zd) < tol:
                return 0
            else:
                return float("inf")
        else:
            return abs(zp - zd) / abs(zp)

    def _get_sense(self) -> str:
        for obj in self.inner.component_objects(Objective):
            sense = obj.sense
            if sense == pyomo.core.kernel.objective.minimize:
                return "min"
            elif sense == pyomo.core.kernel.objective.maximize:
                return "max"
            else:
                raise Exception(f"Unknown sense: ${sense}")
        raise Exception(f"No objective")

    def write(self, filename: str) -> None:
        self.inner.write(filename, io_options={"symbolic_solver_labels": True})
