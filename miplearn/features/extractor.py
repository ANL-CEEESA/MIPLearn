#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import collections
import numbers
from math import log, isfinite
from typing import TYPE_CHECKING, Dict, Optional, List, Any, Tuple, KeysView, cast

import numpy as np

from miplearn.features.sample import Sample

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
        sample.put_array("static_var_lower_bounds", variables.lower_bounds)
        sample.put_array("static_var_names", variables.names)
        sample.put_array("static_var_obj_coeffs", variables.obj_coeffs)
        sample.put_array("static_var_types", variables.types)
        sample.put_array("static_var_upper_bounds", variables.upper_bounds)
        sample.put_vector("static_constr_names", constraints.names)
        # sample.put("static_constr_lhs", constraints.lhs)
        sample.put_array("static_constr_rhs", constraints.rhs)
        sample.put_vector("static_constr_senses", constraints.senses)
        vars_features_user, var_categories = self._extract_user_features_vars(
            instance, sample
        )
        sample.put_vector("static_var_categories", var_categories)
        self._extract_user_features_constrs(instance, sample)
        self._extract_user_features_instance(instance, sample)
        alw17 = self._extract_var_features_AlvLouWeh2017(sample)
        sample.put_vector_list(
            "static_var_features",
            self._combine(
                [
                    alw17,
                    vars_features_user,
                    sample.get_array("static_var_lower_bounds"),
                    sample.get_array("static_var_obj_coeffs"),
                    sample.get_array("static_var_upper_bounds"),
                ],
            ),
        )

    def extract_after_lp_features(
        self,
        solver: "InternalSolver",
        sample: Sample,
    ) -> None:
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
        sample.put_vector("lp_constr_basis_status", constraints.basis_status)
        sample.put_array("lp_constr_dual_values", constraints.dual_values)
        sample.put_array("lp_constr_sa_rhs_down", constraints.sa_rhs_down)
        sample.put_array("lp_constr_sa_rhs_up", constraints.sa_rhs_up)
        sample.put_array("lp_constr_slacks", constraints.slacks)
        alw17 = self._extract_var_features_AlvLouWeh2017(sample)
        sample.put_vector_list(
            "lp_var_features",
            self._combine(
                [
                    alw17,
                    sample.get_array("lp_var_reduced_costs"),
                    sample.get_array("lp_var_sa_lb_down"),
                    sample.get_array("lp_var_sa_lb_up"),
                    sample.get_array("lp_var_sa_obj_down"),
                    sample.get_array("lp_var_sa_obj_up"),
                    sample.get_array("lp_var_sa_ub_down"),
                    sample.get_array("lp_var_sa_ub_up"),
                    sample.get_array("lp_var_values"),
                    sample.get_vector_list("static_var_features"),
                ],
            ),
        )
        sample.put_vector_list(
            "lp_constr_features",
            self._combine(
                [
                    sample.get_vector_list("static_constr_features"),
                    sample.get_array("lp_constr_dual_values"),
                    sample.get_array("lp_constr_sa_rhs_down"),
                    sample.get_array("lp_constr_sa_rhs_up"),
                    sample.get_array("lp_constr_slacks"),
                ],
            ),
        )
        static_instance_features = sample.get_vector("static_instance_features")
        assert static_instance_features is not None
        sample.put_vector(
            "lp_instance_features",
            static_instance_features
            + [
                sample.get_scalar("lp_value"),
                sample.get_scalar("lp_wallclock_time"),
            ],
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

    def _extract_user_features_vars(
        self,
        instance: "Instance",
        sample: Sample,
    ) -> Tuple[List, List]:
        # Query variable names
        var_names = sample.get_array("static_var_names")
        assert var_names is not None

        # Query variable features and categories
        var_features_dict = {
            v.encode(): f for (v, f) in instance.get_variable_features().items()
        }
        var_categories_dict = {
            v.encode(): f for (v, f) in instance.get_variable_categories().items()
        }

        # Assert that variables in user-provided dicts actually exist
        var_names_set = set(var_names)
        for keys in [var_features_dict.keys(), var_categories_dict.keys()]:
            for vn in cast(KeysView, keys):
                assert (
                    vn in var_names_set
                ), f"Variable {vn!r} not found in the problem; {var_names_set}"

        # Assemble into compact lists
        user_features: List[Optional[List[float]]] = []
        categories: List[Optional[str]] = []
        for (i, var_name) in enumerate(var_names):
            if var_name not in var_categories_dict:
                user_features.append(None)
                categories.append(None)
                continue
            category: str = var_categories_dict[var_name]
            assert isinstance(category, str), (
                f"Variable category must be a string. "
                f"Found {type(category).__name__} instead for var={var_name}."
            )
            categories.append(category)
            user_features_i: Optional[List[float]] = None
            if var_name in var_features_dict:
                user_features_i = var_features_dict[var_name]
                if isinstance(user_features_i, np.ndarray):
                    user_features_i = user_features_i.tolist()
                assert isinstance(user_features_i, list), (
                    f"Variable features must be a list. "
                    f"Found {type(user_features_i).__name__} instead for "
                    f"var={var_name}."
                )
                for v in user_features_i:
                    assert isinstance(v, numbers.Real), (
                        f"Variable features must be a list of numbers. "
                        f"Found {type(v).__name__} instead "
                        f"for var={var_name}."
                    )
                user_features_i = list(user_features_i)
            user_features.append(user_features_i)
        return user_features, categories

    def _extract_user_features_constrs(
        self,
        instance: "Instance",
        sample: Sample,
    ) -> None:
        has_static_lazy = instance.has_static_lazy_constraints()
        user_features: List[Optional[List[float]]] = []
        categories: List[Optional[str]] = []
        lazy: List[bool] = []
        constr_categories_dict = instance.get_constraint_categories()
        constr_features_dict = instance.get_constraint_features()
        constr_names = sample.get_vector("static_constr_names")
        assert constr_names is not None

        for (cidx, cname) in enumerate(constr_names):
            category: Optional[str] = cname
            if cname in constr_categories_dict:
                category = constr_categories_dict[cname]
            if category is None:
                user_features.append(None)
                categories.append(None)
                continue
            assert isinstance(category, str), (
                f"Constraint category must be a string. "
                f"Found {type(category).__name__} instead for cname={cname}.",
            )
            categories.append(category)
            cf: Optional[List[float]] = None
            if cname in constr_features_dict:
                cf = constr_features_dict[cname]
                if isinstance(cf, np.ndarray):
                    cf = cf.tolist()
                assert isinstance(cf, list), (
                    f"Constraint features must be a list. "
                    f"Found {type(cf).__name__} instead for cname={cname}."
                )
                for f in cf:
                    assert isinstance(f, numbers.Real), (
                        f"Constraint features must be a list of numbers. "
                        f"Found {type(f).__name__} instead for cname={cname}."
                    )
                cf = list(cf)
            user_features.append(cf)
            if has_static_lazy:
                lazy.append(instance.is_constraint_lazy(cname))
            else:
                lazy.append(False)
        sample.put_vector_list("static_constr_features", user_features)
        sample.put_vector("static_constr_lazy", lazy)
        sample.put_vector("static_constr_categories", categories)

    def _extract_user_features_instance(
        self,
        instance: "Instance",
        sample: Sample,
    ) -> None:
        user_features = instance.get_instance_features()
        if isinstance(user_features, np.ndarray):
            user_features = user_features.tolist()
        assert isinstance(user_features, list), (
            f"Instance features must be a list. "
            f"Found {type(user_features).__name__} instead."
        )
        for v in user_features:
            assert isinstance(v, numbers.Real), (
                f"Instance features must be a list of numbers. "
                f"Found {type(v).__name__} instead."
            )
        constr_lazy = sample.get_vector("static_constr_lazy")
        assert constr_lazy is not None
        sample.put_vector("static_instance_features", user_features)
        sample.put_scalar("static_constr_lazy_count", sum(constr_lazy))

    # Alvarez, A. M., Louveaux, Q., & Wehenkel, L. (2017). A machine learning-based
    # approximation of strong branching. INFORMS Journal on Computing, 29(1), 185-195.
    def _extract_var_features_AlvLouWeh2017(self, sample: Sample) -> List:
        obj_coeffs = sample.get_vector("static_var_obj_coeffs")
        obj_sa_down = sample.get_vector("lp_var_sa_obj_down")
        obj_sa_up = sample.get_vector("lp_var_sa_obj_up")
        values = sample.get_vector(f"lp_var_values")
        assert obj_coeffs is not None

        pos_obj_coeff_sum = 0.0
        neg_obj_coeff_sum = 0.0
        for coeff in obj_coeffs:
            if coeff > 0:
                pos_obj_coeff_sum += coeff
            if coeff < 0:
                neg_obj_coeff_sum += -coeff

        features = []
        for i in range(len(obj_coeffs)):
            f: List[float] = []
            if obj_coeffs is not None:
                # Feature 1
                f.append(np.sign(obj_coeffs[i]))

                # Feature 2
                if pos_obj_coeff_sum > 0:
                    f.append(abs(obj_coeffs[i]) / pos_obj_coeff_sum)
                else:
                    f.append(0.0)

                # Feature 3
                if neg_obj_coeff_sum > 0:
                    f.append(abs(obj_coeffs[i]) / neg_obj_coeff_sum)
                else:
                    f.append(0.0)

            if values is not None:
                # Feature 37
                f.append(
                    min(
                        values[i] - np.floor(values[i]),
                        np.ceil(values[i]) - values[i],
                    )
                )

            if obj_sa_up is not None:
                assert obj_sa_down is not None
                assert obj_coeffs is not None

                # Convert inf into large finite numbers
                sd = max(-1e20, obj_sa_down[i])
                su = min(1e20, obj_sa_up[i])
                obj = obj_coeffs[i]

                # Features 44 and 46
                f.append(np.sign(obj_sa_up[i]))
                f.append(np.sign(obj_sa_down[i]))

                # Feature 47
                csign = np.sign(obj)
                if csign != 0 and ((obj - sd) / csign) > 0.001:
                    f.append(log((obj - sd) / csign))
                else:
                    f.append(0.0)

                # Feature 48
                if csign != 0 and ((su - obj) / csign) > 0.001:
                    f.append(log((su - obj) / csign))
                else:
                    f.append(0.0)

            for (i, v) in enumerate(f):
                if not isfinite(v):
                    f[i] = 0.0

            features.append(f)
        return features

    def _combine(
        self,
        items: List,
    ) -> List[List[float]]:
        combined: List[List[float]] = []
        for series in items:
            if series is None:
                continue
            if len(combined) == 0:
                for i in range(len(series)):
                    combined.append([])
            for (i, s) in enumerate(series):
                if s is None:
                    continue
                elif isinstance(s, list):
                    combined[i].extend([_clip(sj) for sj in s])
                else:
                    combined[i].append(_clip(s))
        return combined


def _clip(vi: float) -> float:
    if not isfinite(vi):
        return max(min(vi, 1e20), -1e20)
    return vi
