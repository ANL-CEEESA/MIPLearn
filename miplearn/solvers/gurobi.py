#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import json
from typing import Dict, Optional, Callable, Any, List

import gurobipy as gp
from gurobipy import GRB, GurobiError
import numpy as np
from scipy.sparse import lil_matrix

from miplearn.h5 import H5File
from miplearn.solvers.abstract import AbstractModel

logger = logging.getLogger(__name__)


def _gurobi_callback(model: AbstractModel, gp_model: gp.Model, where: int) -> None:
    assert isinstance(gp_model, gp.Model)

    # Lazy constraints
    if model.lazy_separate is not None:
        assert model.lazy_enforce is not None
        assert model.lazy_ is not None
        if where == GRB.Callback.MIPSOL:
            model.where = model.WHERE_LAZY
            violations = model.lazy_separate(model)
            if len(violations) > 0:
                model.lazy_.extend(violations)
                model.lazy_enforce(model, violations)

    # User cuts
    if model.cuts_separate is not None:
        assert model.cuts_enforce is not None
        assert model.cuts_ is not None
        if where == GRB.Callback.MIPNODE:
            status = gp_model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status == GRB.OPTIMAL:
                model.where = model.WHERE_CUTS
                if model.cuts_aot_ is not None:
                    violations = model.cuts_aot_
                    model.cuts_aot_ = None
                    logger.info(f"Enforcing {len(violations)} cuts ahead-of-time...")
                else:
                    violations = model.cuts_separate(model)
                if len(violations) > 0:
                    model.cuts_.extend(violations)
                    model.cuts_enforce(model, violations)

    # Cleanup
    model.where = model.WHERE_DEFAULT


def _gurobi_add_constr(gp_model: gp.Model, where: str, constr: Any) -> None:
    if where == AbstractModel.WHERE_LAZY:
        gp_model.cbLazy(constr)
    elif where == AbstractModel.WHERE_CUTS:
        gp_model.cbCut(constr)
    else:
        gp_model.addConstr(constr)


def _gurobi_set_required_params(model: AbstractModel, gp_model: gp.Model) -> None:
    # Required parameters for lazy constraints
    if model.lazy_enforce is not None:
        gp_model.setParam("PreCrush", 1)
        gp_model.setParam("LazyConstraints", 1)
    # Required parameters for user cuts
    if model.cuts_enforce is not None:
        gp_model.setParam("PreCrush", 1)


class GurobiModel(AbstractModel):
    _supports_basis_status = True
    _supports_sensitivity_analysis = True
    _supports_node_count = True
    _supports_solution_pool = True

    def __init__(
        self,
        inner: gp.Model,
        lazy_separate: Optional[Callable] = None,
        lazy_enforce: Optional[Callable] = None,
        cuts_separate: Optional[Callable] = None,
        cuts_enforce: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.lazy_separate = lazy_separate
        self.lazy_enforce = lazy_enforce
        self.cuts_separate = cuts_separate
        self.cuts_enforce = cuts_enforce
        self.inner = inner

    def add_constrs(
        self,
        var_names: np.ndarray,
        constrs_lhs: np.ndarray,
        constrs_sense: np.ndarray,
        constrs_rhs: np.ndarray,
        stats: Optional[Dict] = None,
    ) -> None:
        assert len(var_names.shape) == 1
        nvars = len(var_names)
        assert len(constrs_lhs.shape) == 2
        nconstrs = constrs_lhs.shape[0]
        assert constrs_lhs.shape[1] == nvars
        assert constrs_sense.shape == (nconstrs,)
        assert constrs_rhs.shape == (nconstrs,)

        gp_vars = [self.inner.getVarByName(var_name.decode()) for var_name in var_names]
        self.inner.addMConstr(constrs_lhs, gp_vars, constrs_sense, constrs_rhs)

        if stats is not None:
            if "Added constraints" not in stats:
                stats["Added constraints"] = 0
            stats["Added constraints"] += nconstrs

    def add_constr(self, constr: Any) -> None:
        _gurobi_add_constr(self.inner, self.where, constr)

    def extract_after_load(self, h5: H5File) -> None:
        """
        Given a model that has just been loaded, extracts static problem
        features, such as variable names and types, objective coefficients, etc.
        """
        self.inner.update()
        self._extract_after_load_vars(h5)
        self._extract_after_load_constrs(h5)
        h5.put_scalar("static_sense", "min" if self.inner.modelSense > 0 else "max")
        h5.put_scalar("static_obj_offset", self.inner.objCon)

    def extract_after_lp(self, h5: H5File) -> None:
        """
        Given a linear programming model that has just been solved, extracts
        dynamic problem features, such as optimal LP solution, basis status,
        etc.
        """
        self._extract_after_lp_vars(h5)
        self._extract_after_lp_constrs(h5)
        h5.put_scalar("lp_obj_value", self.inner.objVal)
        h5.put_scalar("lp_wallclock_time", self.inner.runtime)

    def extract_after_mip(self, h5: H5File) -> None:
        """
        Given a mixed-integer linear programming model that has just been
        solved, extracts dynamic problem features, such as optimal MIP solution.
        """
        h5.put_scalar("mip_wallclock_time", self.inner.runtime)
        h5.put_scalar("mip_node_count", self.inner.nodeCount)
        if self.inner.status == GRB.INFEASIBLE:
            return
        gp_vars = self.inner.getVars()
        gp_constrs = self.inner.getConstrs()
        h5.put_array(
            "mip_var_values",
            np.array(self.inner.getAttr("x", gp_vars), dtype=float),
        )
        h5.put_array(
            "mip_constr_slacks",
            np.abs(np.array(self.inner.getAttr("slack", gp_constrs), dtype=float)),
        )
        h5.put_scalar("mip_obj_value", self.inner.objVal)
        h5.put_scalar("mip_obj_bound", self.inner.objBound)
        try:
            h5.put_scalar("mip_gap", self.inner.mipGap)
        except AttributeError:
            pass
        self._extract_after_mip_solution_pool(h5)
        if self.lazy_ is not None:
            h5.put_scalar("mip_lazy", json.dumps(self.lazy_))
        if self.cuts_ is not None:
            h5.put_scalar("mip_cuts", json.dumps(self.cuts_))

    def fix_variables(
        self,
        var_names: np.ndarray,
        var_values: np.ndarray,
        stats: Optional[Dict] = None,
    ) -> None:
        assert len(var_values.shape) == 1
        assert len(var_values.shape) == 1
        assert var_names.shape == var_values.shape

        n_fixed = 0
        for (var_idx, var_name) in enumerate(var_names):
            var_val = var_values[var_idx]
            if np.isfinite(var_val):
                var = self.inner.getVarByName(var_name.decode())
                var.vtype = "C"
                var.lb = var_val
                var.ub = var_val
                n_fixed += 1
        if stats is not None:
            stats["Fixed variables"] = n_fixed

    def optimize(self) -> None:
        self.lazy_ = []
        self.cuts_ = []

        def callback(_: gp.Model, where: int) -> None:
            _gurobi_callback(self, self.inner, where)

        _gurobi_set_required_params(self, self.inner)

        if self.lazy_enforce is not None or self.cuts_enforce is not None:
            self.inner.optimize(callback)
        else:
            self.inner.optimize()

    def relax(self) -> "GurobiModel":
        return GurobiModel(self.inner.relax())

    def set_time_limit(self, time_limit_sec: float) -> None:
        self.inner.params.timeLimit = time_limit_sec

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

        self.inner.numStart = n_starts
        for start_idx in range(n_starts):
            self.inner.params.startNumber = start_idx
            for (var_idx, var_name) in enumerate(var_names):
                var_val = var_values[start_idx, var_idx]
                if np.isfinite(var_val):
                    var = self.inner.getVarByName(var_name.decode())
                    var.start = var_val

        if stats is not None:
            stats["WS: Count"] = n_starts
            stats["WS: Number of variables set"] = (
                np.isfinite(var_values).mean(axis=0).sum()
            )

    def _extract_after_load_vars(self, h5: H5File) -> None:
        gp_vars = self.inner.getVars()
        for (h5_field, gp_field) in {
            "static_var_names": "varName",
            "static_var_types": "vtype",
        }.items():
            h5.put_array(
                h5_field, np.array(self.inner.getAttr(gp_field, gp_vars), dtype="S")
            )
        for (h5_field, gp_field) in {
            "static_var_upper_bounds": "ub",
            "static_var_lower_bounds": "lb",
            "static_var_obj_coeffs": "obj",
        }.items():
            h5.put_array(
                h5_field, np.array(self.inner.getAttr(gp_field, gp_vars), dtype=float)
            )

    def _extract_after_load_constrs(self, h5: H5File) -> None:
        gp_constrs = self.inner.getConstrs()
        gp_vars = self.inner.getVars()
        rhs = np.array(self.inner.getAttr("rhs", gp_constrs), dtype=float)
        senses = np.array(self.inner.getAttr("sense", gp_constrs), dtype="S")
        names = np.array(self.inner.getAttr("constrName", gp_constrs), dtype="S")
        nrows, ncols = len(gp_constrs), len(gp_vars)
        tmp = lil_matrix((nrows, ncols), dtype=float)
        for (i, gp_constr) in enumerate(gp_constrs):
            expr = self.inner.getRow(gp_constr)
            for j in range(expr.size()):
                tmp[i, expr.getVar(j).index] = expr.getCoeff(j)
        lhs = tmp.tocoo()

        h5.put_array("static_constr_names", names)
        h5.put_array("static_constr_rhs", rhs)
        h5.put_array("static_constr_sense", senses)
        h5.put_sparse("static_constr_lhs", lhs)

    def _extract_after_lp_vars(self, h5: H5File) -> None:
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
                raise Exception(f"unknown vbasis: {b}")

        gp_vars = self.inner.getVars()
        h5.put_array(
            "lp_var_basis_status",
            np.array(
                [
                    _parse_gurobi_vbasis(b)
                    for b in self.inner.getAttr("vbasis", gp_vars)
                ],
                dtype="S",
            ),
        )
        for (h5_field, gp_field) in {
            "lp_var_reduced_costs": "rc",
            "lp_var_sa_obj_up": "saobjUp",
            "lp_var_sa_obj_down": "saobjLow",
            "lp_var_sa_ub_up": "saubUp",
            "lp_var_sa_ub_down": "saubLow",
            "lp_var_sa_lb_up": "salbUp",
            "lp_var_sa_lb_down": "salbLow",
            "lp_var_values": "x",
        }.items():
            h5.put_array(
                h5_field,
                np.array(self.inner.getAttr(gp_field, gp_vars), dtype=float),
            )

    def _extract_after_lp_constrs(self, h5: H5File) -> None:
        def _parse_gurobi_cbasis(v: int) -> str:
            if v == 0:
                return "B"
            if v == -1:
                return "N"
            raise Exception(f"unknown cbasis: {v}")

        gp_constrs = self.inner.getConstrs()
        h5.put_array(
            "lp_constr_basis_status",
            np.array(
                [
                    _parse_gurobi_cbasis(c)
                    for c in self.inner.getAttr("cbasis", gp_constrs)
                ],
                dtype="S",
            ),
        )
        for (h5_field, gp_field) in {
            "lp_constr_dual_values": "pi",
            "lp_constr_sa_rhs_up": "saRhsUp",
            "lp_constr_sa_rhs_down": "saRhsLow",
        }.items():
            h5.put_array(
                h5_field,
                np.array(self.inner.getAttr(gp_field, gp_constrs), dtype=float),
            )
        h5.put_array(
            "lp_constr_slacks",
            np.abs(np.array(self.inner.getAttr("slack", gp_constrs), dtype=float)),
        )

    def _extract_after_mip_solution_pool(self, h5: H5File) -> None:
        gp_vars = self.inner.getVars()
        pool_var_values = []
        pool_obj_values = []
        for i in range(self.inner.SolCount):
            self.inner.params.SolutionNumber = i
            try:
                pool_var_values.append(self.inner.getAttr("Xn", gp_vars))
                pool_obj_values.append(self.inner.PoolObjVal)
            except GurobiError:
                pass
        h5.put_array("pool_var_values", np.array(pool_var_values))
        h5.put_array("pool_obj_values", np.array(pool_obj_values))

    def write(self, filename: str) -> None:
        self.inner.update()
        self.inner.write(filename)
