#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from math import log, isfinite
from typing import TYPE_CHECKING, List, Tuple, Optional

import numpy as np
from scipy.sparse import coo_matrix

from miplearn.features.sample import Sample
from miplearn.solvers.internal import LPSolveStats

if TYPE_CHECKING:
    from miplearn.solvers.internal import InternalSolver
    from miplearn.instance.base import Instance


# noinspection PyPep8Naming
class FeaturesExtractor:
    def __init__(
        self,
        with_sa: bool = True,
        with_lhs: bool = True,
    ) -> None:
        self.with_sa = with_sa
        self.with_lhs = with_lhs
        self.var_features_user: Optional[np.ndarray] = None

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
        sample.put_sparse("static_constr_lhs", constraints.lhs)
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
        self.var_features_user = vars_features_user
        sample.put_array("static_var_categories", var_categories)
        assert variables.lower_bounds is not None
        assert variables.obj_coeffs is not None
        assert variables.upper_bounds is not None
        sample.put_array(
            "static_var_features",
            np.hstack(
                [
                    vars_features_user,
                    self._compute_AlvLouWeh2017(
                        A=constraints.lhs,
                        b=constraints.rhs,
                        c=variables.obj_coeffs,
                    ),
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
            self.var_features_user,
            self._compute_AlvLouWeh2017(
                A=sample.get_sparse("static_constr_lhs"),
                b=sample.get_array("static_constr_rhs"),
                c=sample.get_array("static_var_obj_coeffs"),
                c_sa_up=variables.sa_obj_up,
                c_sa_down=variables.sa_obj_down,
                values=variables.values,
            ),
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

    @classmethod
    def _compute_AlvLouWeh2017(
        cls,
        A: Optional[coo_matrix] = None,
        b: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        c_sa_down: Optional[np.ndarray] = None,
        c_sa_up: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes static variable features described in:
            Alvarez, A. M., Louveaux, Q., & Wehenkel, L. (2017). A machine learning-based
            approximation of strong branching. INFORMS Journal on Computing, 29(1),
            185-195.
        """
        assert b is not None
        assert c is not None
        nvars = len(c)

        c_pos_sum = c[c > 0].sum()
        c_neg_sum = -c[c < 0].sum()

        curr = 0
        max_n_features = 30
        features = np.zeros((nvars, max_n_features))

        def push(v: np.ndarray) -> None:
            nonlocal curr
            features[:, curr] = v
            curr += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            # Feature 1
            push(np.sign(c))

            # Feature 2
            push(np.abs(c) / c_pos_sum)

            # Feature 3
            push(np.abs(c) / c_neg_sum)

            if A is not None:
                M1 = A.T.multiply(1.0 / np.abs(b)).T.tocsr()
                M1_pos = M1[b > 0, :]
                if M1_pos.shape[0] > 0:
                    M1_pos_max = M1_pos.max(axis=0).todense()
                    M1_pos_min = M1_pos.min(axis=0).todense()
                else:
                    M1_pos_max = np.zeros(nvars)
                    M1_pos_min = np.zeros(nvars)
                M1_neg = M1[b < 0, :]
                if M1_neg.shape[0] > 0:
                    M1_neg_max = M1_neg.max(axis=0).todense()
                    M1_neg_min = M1_neg.min(axis=0).todense()
                else:
                    M1_neg_max = np.zeros(nvars)
                    M1_neg_min = np.zeros(nvars)

                # Features 4-11
                push(np.sign(M1_pos_min))
                push(np.sign(M1_pos_max))
                push(np.abs(M1_pos_min))
                push(np.abs(M1_pos_max))
                push(np.sign(M1_neg_min))
                push(np.sign(M1_neg_max))
                push(np.abs(M1_neg_min))
                push(np.abs(M1_neg_max))

            # Feature 37
            if values is not None:
                push(
                    np.minimum(
                        values - np.floor(values),
                        np.ceil(values) - values,
                    )
                )

            # Feature 44
            if c_sa_up is not None:
                push(np.sign(c_sa_up))

            # Feature 46
            if c_sa_down is not None:
                push(np.sign(c_sa_down))

            # Feature 47
            if c_sa_down is not None:
                push(np.log(c - c_sa_down / np.sign(c)))

            # Feature 48
            if c_sa_up is not None:
                push(np.log(c - c_sa_up / np.sign(c)))

        features = features[:, 0:curr]
        _fix_infinity(features)
        return features


def _fix_infinity(m: Optional[np.ndarray]) -> None:
    if m is None:
        return
    masked = np.ma.masked_invalid(m)
    max_values = np.max(masked, axis=0)
    min_values = np.min(masked, axis=0)
    m[:] = np.maximum(np.minimum(m, max_values), min_values)
    m[np.isnan(m)] = 0.0
