#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import collections
import numbers
from dataclasses import dataclass
from math import log, isfinite
from typing import TYPE_CHECKING, Dict, Optional, List, Hashable, Tuple

import numpy as np

if TYPE_CHECKING:
    from miplearn.solvers.internal import InternalSolver, LPSolveStats, MIPSolveStats
    from miplearn.instance.base import Instance


@dataclass
class InstanceFeatures:
    user_features: Optional[List[float]] = None
    lazy_constraint_count: int = 0

    def to_list(self) -> List[float]:
        features: List[float] = []
        if self.user_features is not None:
            features.extend(self.user_features)
        _clip(features)
        return features


@dataclass
class VariableFeatures:
    names: Optional[Tuple[str, ...]] = None
    basis_status: Optional[Tuple[str, ...]] = None
    categories: Optional[Tuple[Optional[Hashable], ...]] = None
    lower_bounds: Optional[Tuple[float, ...]] = None
    obj_coeffs: Optional[Tuple[float, ...]] = None
    reduced_costs: Optional[Tuple[float, ...]] = None
    sa_lb_down: Optional[Tuple[float, ...]] = None
    sa_lb_up: Optional[Tuple[float, ...]] = None
    sa_obj_down: Optional[Tuple[float, ...]] = None
    sa_obj_up: Optional[Tuple[float, ...]] = None
    sa_ub_down: Optional[Tuple[float, ...]] = None
    sa_ub_up: Optional[Tuple[float, ...]] = None
    types: Optional[Tuple[str, ...]] = None
    upper_bounds: Optional[Tuple[float, ...]] = None
    user_features: Optional[Tuple[Optional[Tuple[float, ...]], ...]] = None
    values: Optional[Tuple[float, ...]] = None

    # Alvarez, A. M., Louveaux, Q., & Wehenkel, L. (2017). A machine learning-based
    # approximation of strong branching. INFORMS Journal on Computing, 29(1), 185-195.
    alvarez_2017: Optional[List[List[float]]] = None

    def to_list(self, index: int) -> List[float]:
        features: List[float] = []
        for attr in [
            "lower_bounds",
            "obj_coeffs",
            "reduced_costs",
            "sa_lb_down",
            "sa_lb_up",
            "sa_obj_down",
            "sa_obj_up",
            "sa_ub_down",
            "sa_ub_up",
            "upper_bounds",
            "values",
        ]:
            if getattr(self, attr) is not None:
                features.append(getattr(self, attr)[index])
        for attr in ["user_features", "alvarez_2017"]:
            if getattr(self, attr) is not None:
                if getattr(self, attr)[index] is not None:
                    features.extend(getattr(self, attr)[index])
        _clip(features)
        return features


@dataclass
class ConstraintFeatures:
    basis_status: Optional[Tuple[str, ...]] = None
    categories: Optional[Tuple[Optional[Hashable], ...]] = None
    dual_values: Optional[Tuple[float, ...]] = None
    names: Optional[Tuple[str, ...]] = None
    lazy: Optional[Tuple[bool, ...]] = None
    lhs: Optional[Tuple[Tuple[Tuple[str, float], ...], ...]] = None
    rhs: Optional[Tuple[float, ...]] = None
    sa_rhs_down: Optional[Tuple[float, ...]] = None
    sa_rhs_up: Optional[Tuple[float, ...]] = None
    senses: Optional[Tuple[str, ...]] = None
    slacks: Optional[Tuple[float, ...]] = None
    user_features: Optional[Tuple[Optional[Tuple[float, ...]], ...]] = None

    def to_list(self, index: int) -> List[float]:
        features: List[float] = []
        for attr in [
            "dual_values",
            "rhs",
            "slacks",
        ]:
            if getattr(self, attr) is not None:
                features.append(getattr(self, attr)[index])
        for attr in ["user_features"]:
            if getattr(self, attr) is not None:
                if getattr(self, attr)[index] is not None:
                    features.extend(getattr(self, attr)[index])
        _clip(features)
        return features

    def __getitem__(self, selected: Tuple[bool, ...]) -> "ConstraintFeatures":
        return ConstraintFeatures(
            basis_status=self._filter(self.basis_status, selected),
            categories=self._filter(self.categories, selected),
            dual_values=self._filter(self.dual_values, selected),
            names=self._filter(self.names, selected),
            lazy=self._filter(self.lazy, selected),
            lhs=self._filter(self.lhs, selected),
            rhs=self._filter(self.rhs, selected),
            sa_rhs_down=self._filter(self.sa_rhs_down, selected),
            sa_rhs_up=self._filter(self.sa_rhs_up, selected),
            senses=self._filter(self.senses, selected),
            slacks=self._filter(self.slacks, selected),
            user_features=self._filter(self.user_features, selected),
        )

    def _filter(
        self,
        obj: Optional[Tuple],
        selected: Tuple[bool, ...],
    ) -> Optional[Tuple]:
        if obj is None:
            return None
        return tuple(obj[i] for (i, selected_i) in enumerate(selected) if selected_i)


@dataclass
class Constraint:
    basis_status: Optional[str] = None
    category: Optional[Hashable] = None
    dual_value: Optional[float] = None
    lazy: bool = False
    lhs: Optional[Dict[str, float]] = None
    rhs: float = 0.0
    sa_rhs_down: Optional[float] = None
    sa_rhs_up: Optional[float] = None
    sense: str = "<"
    slack: Optional[float] = None
    user_features: Optional[List[float]] = None

    def to_list(self) -> List[float]:
        features: List[float] = []
        for attr in [
            "dual value",
            "rhs",
            "sa_rhs_down",
            "sa_rhs_up",
            "slack",
        ]:
            if getattr(self, attr) is not None:
                features.append(getattr(self, attr))
        for attr in ["user_features"]:
            if getattr(self, attr) is not None:
                features.extend(getattr(self, attr))
        if self.lhs is not None and len(self.lhs) > 0:
            features.append(np.max(self.lhs.values()))
            features.append(np.average(self.lhs.values()))
            features.append(np.min(self.lhs.values()))
        _clip(features)
        return features


@dataclass
class Features:
    instance: Optional[InstanceFeatures] = None
    variables: Optional[VariableFeatures] = None
    constraints: Optional[ConstraintFeatures] = None
    lp_solve: Optional["LPSolveStats"] = None
    mip_solve: Optional["MIPSolveStats"] = None
    extra: Optional[Dict] = None


@dataclass
class Sample:
    after_load: Optional[Features] = None
    after_lp: Optional[Features] = None
    after_mip: Optional[Features] = None


class FeaturesExtractor:
    def __init__(
        self,
        with_sa: bool = True,
        with_lhs: bool = True,
    ) -> None:
        self.with_sa = with_sa
        self.with_lhs = with_lhs

    def extract(
        self,
        instance: "Instance",
        solver: "InternalSolver",
        with_static: bool = True,
    ) -> Features:
        features = Features()
        features.variables = solver.get_variables(
            with_static=with_static,
            with_sa=self.with_sa,
        )
        features.constraints = solver.get_constraints(
            with_static=with_static,
            with_sa=self.with_sa,
            with_lhs=self.with_lhs,
        )
        if with_static:
            self._extract_user_features_vars(instance, features)
            self._extract_user_features_constrs(instance, features)
            self._extract_user_features_instance(instance, features)
            self._extract_alvarez_2017(features)
        return features

    def _extract_user_features_vars(
        self,
        instance: "Instance",
        features: Features,
    ) -> None:
        assert features.variables is not None
        assert features.variables.names is not None
        categories: List[Hashable] = []
        user_features: List[Optional[Tuple[float, ...]]] = []
        for (i, var_name) in enumerate(features.variables.names):
            category: Hashable = instance.get_variable_category(var_name)
            user_features_i: Optional[List[float]] = None
            if category is not None:
                assert isinstance(category, collections.Hashable), (
                    f"Variable category must be be hashable. "
                    f"Found {type(category).__name__} instead for var={var_name}."
                )
                user_features_i = instance.get_variable_features(var_name)
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
            categories.append(category)
            if user_features_i is None:
                user_features.append(None)
            else:
                user_features.append(tuple(user_features_i))
        features.variables.categories = tuple(categories)
        features.variables.user_features = tuple(user_features)

    def _extract_user_features_constrs(
        self,
        instance: "Instance",
        features: Features,
    ) -> None:
        assert features.constraints is not None
        assert features.constraints.names is not None
        has_static_lazy = instance.has_static_lazy_constraints()
        user_features: List[Optional[Tuple[float, ...]]] = []
        categories: List[Optional[Hashable]] = []
        lazy: List[bool] = []
        for (cidx, cname) in enumerate(features.constraints.names):
            cf: Optional[List[float]] = None
            category: Optional[Hashable] = instance.get_constraint_category(cname)
            if category is not None:
                categories.append(category)
                assert isinstance(category, collections.Hashable), (
                    f"Constraint category must be hashable. "
                    f"Found {type(category).__name__} instead for cname={cname}.",
                )
                cf = instance.get_constraint_features(cname)
                if isinstance(cf, np.ndarray):
                    cf = tuple(cf.tolist())
                assert isinstance(cf, list), (
                    f"Constraint features must be a list. "
                    f"Found {type(cf).__name__} instead for cname={cname}."
                )
                for f in cf:
                    assert isinstance(f, numbers.Real), (
                        f"Constraint features must be a list of numbers. "
                        f"Found {type(f).__name__} instead for cname={cname}."
                    )
                user_features.append(tuple(cf))
            else:
                user_features.append(None)
                categories.append(None)
            if has_static_lazy:
                lazy.append(instance.is_constraint_lazy(cname))
            else:
                lazy.append(False)
        features.constraints.user_features = tuple(user_features)
        features.constraints.lazy = tuple(lazy)
        features.constraints.categories = tuple(categories)

    def _extract_user_features_instance(
        self,
        instance: "Instance",
        features: Features,
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
        assert features.constraints is not None
        assert features.constraints.lazy is not None
        features.instance = InstanceFeatures(
            user_features=user_features,
            lazy_constraint_count=sum(features.constraints.lazy),
        )

    def _extract_alvarez_2017(self, features: Features) -> None:
        assert features.variables is not None
        assert features.variables.names is not None

        obj_coeffs = features.variables.obj_coeffs
        obj_sa_down = features.variables.sa_obj_down
        obj_sa_up = features.variables.sa_obj_up
        values = features.variables.values

        pos_obj_coeff_sum = 0.0
        neg_obj_coeff_sum = 0.0
        if obj_coeffs is not None:
            for coeff in obj_coeffs:
                if coeff > 0:
                    pos_obj_coeff_sum += coeff
                if coeff < 0:
                    neg_obj_coeff_sum += -coeff

        features.variables.alvarez_2017 = []
        for i in range(len(features.variables.names)):
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

            for v in f:
                assert isfinite(v), f"non-finite elements detected: {f}"
            features.variables.alvarez_2017.append(f)


def _clip(v: List[float]) -> None:
    for (i, vi) in enumerate(v):
        if not isfinite(vi):
            v[i] = max(min(vi, 1e20), -1e20)
