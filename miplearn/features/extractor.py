#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from math import log, isfinite
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from miplearn.features.sample import Sample
from miplearn.solvers.internal import LPSolveStats

if TYPE_CHECKING:
    from miplearn.solvers.internal import InternalSolver
    from miplearn.instance.base import Instance


class FeaturesExtractor:
    def __init__(
        self,
        with_sa: bool = True,
        with_lhs: bool = True,
    ) -> None:
        self.with_sa = with_sa
        self.with_lhs = with_lhs

    def extract_after_load_features(
        self,
        instance: "Instance",
        solver: "InternalSolver",
        sample: Sample,
    ) -> None:
        variables = solver.get_variables(with_static=True)
        constraints = solver.get_constraints(with_static=True, with_lhs=self.with_lhs)
        assert constraints.names is not None
        sample.put_array("static_var_lower_bounds", variables.lower_bounds)
        sample.put_array("static_var_names", variables.names)
        sample.put_array("static_var_obj_coeffs", variables.obj_coeffs)
        sample.put_array("static_var_types", variables.types)
        sample.put_array("static_var_upper_bounds", variables.upper_bounds)
        sample.put_array("static_constr_names", constraints.names)
        # sample.put("static_constr_lhs", constraints.lhs)
        sample.put_array("static_constr_rhs", constraints.rhs)
        sample.put_array("static_constr_senses", constraints.senses)

        # Instance features
        self._extract_user_features_instance(instance, sample)

        # Constraint features
        (
            constr_features,
            constr_categories,
            constr_lazy,
        ) = FeaturesExtractor._extract_user_features_constrs(
            instance,
            constraints.names,
        )
        sample.put_array("static_constr_features", constr_features)
        sample.put_array("static_constr_categories", constr_categories)
        sample.put_array("static_constr_lazy", constr_lazy)
        sample.put_scalar("static_constr_lazy_count", int(constr_lazy.sum()))

        # Variable features
        (
            vars_features_user,
            var_categories,
        ) = self._extract_user_features_vars(instance, sample)
        sample.put_array("static_var_categories", var_categories)
        assert variables.lower_bounds is not None
        assert variables.obj_coeffs is not None
        assert variables.upper_bounds is not None
        sample.put_array(
            "static_var_features",
            np.hstack(
                [
                    vars_features_user,
                    self._extract_var_features_AlvLouWeh2017(sample),
                    variables.lower_bounds.reshape(-1, 1),
                    variables.obj_coeffs.reshape(-1, 1),
                    variables.upper_bounds.reshape(-1, 1),
                ]
            ),
        )

    def extract_after_lp_features(
        self,
        solver: "InternalSolver",
        sample: Sample,
        lp_stats: LPSolveStats,
    ) -> None:
        for (k, v) in lp_stats.__dict__.items():
            sample.put_scalar(k, v)
        variables = solver.get_variables(with_static=False, with_sa=self.with_sa)
        constraints = solver.get_constraints(with_static=False, with_sa=self.with_sa)
        sample.put_array("lp_var_basis_status", variables.basis_status)
        sample.put_array("lp_var_reduced_costs", variables.reduced_costs)
        sample.put_array("lp_var_sa_lb_down", variables.sa_lb_down)
        sample.put_array("lp_var_sa_lb_up", variables.sa_lb_up)
        sample.put_array("lp_var_sa_obj_down", variables.sa_obj_down)
        sample.put_array("lp_var_sa_obj_up", variables.sa_obj_up)
        sample.put_array("lp_var_sa_ub_down", variables.sa_ub_down)
        sample.put_array("lp_var_sa_ub_up", variables.sa_ub_up)
        sample.put_array("lp_var_values", variables.values)
        sample.put_array("lp_constr_basis_status", constraints.basis_status)
        sample.put_array("lp_constr_dual_values", constraints.dual_values)
        sample.put_array("lp_constr_sa_rhs_down", constraints.sa_rhs_down)
        sample.put_array("lp_constr_sa_rhs_up", constraints.sa_rhs_up)
        sample.put_array("lp_constr_slacks", constraints.slacks)

        # Variable features
        lp_var_features_list = []
        for f in [
            sample.get_array("static_var_features"),
            self._extract_var_features_AlvLouWeh2017(sample),
        ]:
            if f is not None:
                lp_var_features_list.append(f)
        for f in [
            variables.reduced_costs,
            variables.sa_lb_down,
            variables.sa_lb_up,
            variables.sa_obj_down,
            variables.sa_obj_up,
            variables.sa_ub_down,
            variables.sa_ub_up,
            variables.values,
        ]:
            if f is not None:
                lp_var_features_list.append(f.reshape(-1, 1))
        lp_var_features = np.hstack(lp_var_features_list)
        _fix_infinity(lp_var_features)
        sample.put_array("lp_var_features", lp_var_features)

        # Constraint features
        lp_constr_features_list = []
        for f in [sample.get_array("static_constr_features")]:
            if f is not None:
                lp_constr_features_list.append(f)
        for f in [
            sample.get_array("lp_constr_dual_values"),
            sample.get_array("lp_constr_sa_rhs_down"),
            sample.get_array("lp_constr_sa_rhs_up"),
            sample.get_array("lp_constr_slacks"),
        ]:
            if f is not None:
                lp_constr_features_list.append(f.reshape(-1, 1))
        lp_constr_features = np.hstack(lp_constr_features_list)
        _fix_infinity(lp_constr_features)
        sample.put_array("lp_constr_features", lp_constr_features)

        # Build lp_instance_features
        static_instance_features = sample.get_array("static_instance_features")
        assert static_instance_features is not None
        assert lp_stats.lp_value is not None
        assert lp_stats.lp_wallclock_time is not None
        sample.put_array(
            "lp_instance_features",
            np.hstack(
                [
                    static_instance_features,
                    lp_stats.lp_value,
                    lp_stats.lp_wallclock_time,
                ]
            ),
        )

    def extract_after_mip_features(
        self,
        solver: "InternalSolver",
        sample: Sample,
    ) -> None:
        variables = solver.get_variables(with_static=False, with_sa=False)
        constraints = solver.get_constraints(with_static=False, with_sa=False)
        sample.put_array("mip_var_values", variables.values)
        sample.put_array("mip_constr_slacks", constraints.slacks)

    # noinspection DuplicatedCode
    def _extract_user_features_vars(
        self,
        instance: "Instance",
        sample: Sample,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Query variable names
        var_names = sample.get_array("static_var_names")
        assert var_names is not None

        # Query variable features
        var_features = instance.get_variable_features(var_names)
        assert isinstance(var_features, np.ndarray), (
            f"Variable features must be a numpy array. "
            f"Found {var_features.__class__} instead."
        )
        assert len(var_features.shape) == 2, (
            f"Variable features must be 2-dimensional array. "
            f"Found array with shape {var_features.shape} instead."
        )
        assert var_features.shape[0] == len(var_names), (
            f"Variable features must have exactly {len(var_names)} rows. "
            f"Found {var_features.shape[0]} rows instead."
        )
        assert var_features.dtype.kind in ["f"], (
            f"Variable features must be floating point numbers. "
            f"Found {var_features.dtype} instead."
        )

        # Query variable categories
        var_categories = instance.get_variable_categories(var_names)
        assert isinstance(var_categories, np.ndarray), (
            f"Variable categories must be a numpy array. "
            f"Found {var_categories.__class__} instead."
        )
        assert len(var_categories.shape) == 1, (
            f"Variable categories must be a vector. "
            f"Found array with shape {var_categories.shape} instead."
        )
        assert len(var_categories) == len(var_names), (
            f"Variable categories must have exactly {len(var_names)} elements. "
            f"Found {var_categories.shape[0]} elements instead."
        )
        assert var_categories.dtype.kind == "S", (
            f"Variable categories must be a numpy array with dtype='S'. "
            f"Found {var_categories.dtype} instead."
        )
        return var_features, var_categories

    # noinspection DuplicatedCode
    @classmethod
    def _extract_user_features_constrs(
        cls,
        instance: "Instance",
        constr_names: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Query constraint features
        constr_features = instance.get_constraint_features(constr_names)
        assert isinstance(constr_features, np.ndarray), (
            f"get_constraint_features must return a numpy array. "
            f"Found {constr_features.__class__} instead."
        )
        assert len(constr_features.shape) == 2, (
            f"get_constraint_features must return a 2-dimensional array. "
            f"Found array with shape {constr_features.shape} instead."
        )
        assert constr_features.shape[0] == len(constr_names), (
            f"get_constraint_features must return an array with {len(constr_names)} "
            f"rows. Found {constr_features.shape[0]} rows instead."
        )
        assert constr_features.dtype.kind in ["f"], (
            f"get_constraint_features must return floating point numbers. "
            f"Found {constr_features.dtype} instead."
        )

        # Query constraint categories
        constr_categories = instance.get_constraint_categories(constr_names)
        assert isinstance(constr_categories, np.ndarray), (
            f"get_constraint_categories must return a numpy array. "
            f"Found {constr_categories.__class__} instead."
        )
        assert len(constr_categories.shape) == 1, (
            f"get_constraint_categories must return a vector. "
            f"Found array with shape {constr_categories.shape} instead."
        )
        assert len(constr_categories) == len(constr_names), (
            f"get_constraint_categories must return a vector with {len(constr_names)} "
            f"elements. Found {constr_categories.shape[0]} elements instead."
        )
        assert constr_categories.dtype.kind == "S", (
            f"get_constraint_categories must return a numpy array with dtype='S'. "
            f"Found {constr_categories.dtype} instead."
        )

        # Query constraint lazy attribute
        constr_lazy = instance.are_constraints_lazy(constr_names)
        assert isinstance(constr_lazy, np.ndarray), (
            f"are_constraints_lazy must return a numpy array. "
            f"Found {constr_lazy.__class__} instead."
        )
        assert len(constr_lazy.shape) == 1, (
            f"are_constraints_lazy must return a vector. "
            f"Found array with shape {constr_lazy.shape} instead."
        )
        assert constr_lazy.shape[0] == len(constr_names), (
            f"are_constraints_lazy must return a vector with {len(constr_names)} "
            f"elements. Found {constr_lazy.shape[0]} elements instead."
        )
        assert constr_lazy.dtype.kind == "b", (
            f"are_constraints_lazy must return a boolean array. "
            f"Found {constr_lazy.dtype} instead."
        )

        return constr_features, constr_categories, constr_lazy

    def _extract_user_features_instance(
        self,
        instance: "Instance",
        sample: Sample,
    ) -> None:
        features = instance.get_instance_features()
        assert isinstance(features, np.ndarray), (
            f"Instance features must be a numpy array. "
            f"Found {features.__class__} instead."
        )
        assert len(features.shape) == 1, (
            f"Instance features must be a vector. "
            f"Found array with shape {features.shape} instead."
        )
        assert features.dtype.kind in [
            "f"
        ], f"Instance features have unsupported {features.dtype}"
        sample.put_array("static_instance_features", features)

    # Alvarez, A. M., Louveaux, Q., & Wehenkel, L. (2017). A machine learning-based
    # approximation of strong branching. INFORMS Journal on Computing, 29(1), 185-195.
    def _extract_var_features_AlvLouWeh2017(self, sample: Sample) -> np.ndarray:
        obj_coeffs = sample.get_array("static_var_obj_coeffs")
        obj_sa_down = sample.get_array("lp_var_sa_obj_down")
        obj_sa_up = sample.get_array("lp_var_sa_obj_up")
        values = sample.get_array("lp_var_values")

        assert obj_coeffs is not None
        obj_coeffs = obj_coeffs.astype(float)
        _fix_infinity(obj_coeffs)
        nvars = len(obj_coeffs)

        if obj_sa_down is not None:
            obj_sa_down = obj_sa_down.astype(float)
            _fix_infinity(obj_sa_down)

        if obj_sa_up is not None:
            obj_sa_up = obj_sa_up.astype(float)
            _fix_infinity(obj_sa_up)

        if values is not None:
            values = values.astype(float)
            _fix_infinity(values)

        pos_obj_coeffs_sum = obj_coeffs[obj_coeffs > 0].sum()
        neg_obj_coeffs_sum = -obj_coeffs[obj_coeffs < 0].sum()

        curr = 0
        max_n_features = 8
        features = np.zeros((nvars, max_n_features))
        with np.errstate(divide="ignore", invalid="ignore"):
            # Feature 1
            features[:, curr] = np.sign(obj_coeffs)
            curr += 1

            # Feature 2
            if abs(pos_obj_coeffs_sum) > 0:
                features[:, curr] = np.abs(obj_coeffs) / pos_obj_coeffs_sum
                curr += 1

            # Feature 3
            if abs(neg_obj_coeffs_sum) > 0:
                features[:, curr] = np.abs(obj_coeffs) / neg_obj_coeffs_sum
                curr += 1

            # Feature 37
            if values is not None:
                features[:, curr] = np.minimum(
                    values - np.floor(values),
                    np.ceil(values) - values,
                )
                curr += 1

            # Feature 44
            if obj_sa_up is not None:
                features[:, curr] = np.sign(obj_sa_up)
                curr += 1

            # Feature 46
            if obj_sa_down is not None:
                features[:, curr] = np.sign(obj_sa_down)
                curr += 1

            # Feature 47
            if obj_sa_down is not None:
                features[:, curr] = np.log(
                    obj_coeffs - obj_sa_down / np.sign(obj_coeffs)
                )
                curr += 1

            # Feature 48
            if obj_sa_up is not None:
                features[:, curr] = np.log(obj_coeffs - obj_sa_up / np.sign(obj_coeffs))
                curr += 1

        features = features[:, 0:curr]
        _fix_infinity(features)
        return features


def _fix_infinity(m: np.ndarray) -> None:
    masked = np.ma.masked_invalid(m)
    max_values = np.max(masked, axis=0)
    min_values = np.min(masked, axis=0)
    m[:] = np.maximum(np.minimum(m, max_values), min_values)
    m[np.isnan(m)] = 0.0
