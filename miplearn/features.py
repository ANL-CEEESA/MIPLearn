#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import collections
import numbers
from dataclasses import dataclass
from math import log, isfinite
from typing import TYPE_CHECKING, Dict, Optional, List, Hashable, Tuple

import numpy as np

from miplearn.types import Category

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
    categories: Optional[Tuple[Hashable, ...]] = None
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
    user_features: Optional[Tuple[Tuple[float, ...]]] = None
    values: Optional[Tuple[float, ...]] = None


@dataclass
class Variable:
    basis_status: Optional[str] = None
    category: Optional[Hashable] = None
    lower_bound: Optional[float] = None
    obj_coeff: Optional[float] = None
    reduced_cost: Optional[float] = None
    sa_lb_down: Optional[float] = None
    sa_lb_up: Optional[float] = None
    sa_obj_down: Optional[float] = None
    sa_obj_up: Optional[float] = None
    sa_ub_down: Optional[float] = None
    sa_ub_up: Optional[float] = None
    type: Optional[str] = None
    upper_bound: Optional[float] = None
    user_features: Optional[List[float]] = None
    value: Optional[float] = None

    # Alvarez, A. M., Louveaux, Q., & Wehenkel, L. (2017). A machine learning-based
    # approximation of strong branching. INFORMS Journal on Computing, 29(1), 185-195.
    alvarez_2017: Optional[List[float]] = None

    def to_list(self) -> List[float]:
        features: List[float] = []
        for attr in [
            "lower_bound",
            "obj_coeff",
            "reduced_cost",
            "sa_lb_down",
            "sa_lb_up",
            "sa_obj_down",
            "sa_obj_up",
            "sa_ub_down",
            "sa_ub_up",
            "upper_bound",
            "value",
        ]:
            if getattr(self, attr) is not None:
                features.append(getattr(self, attr))
        for attr in ["user_features", "alvarez_2017"]:
            if getattr(self, attr) is not None:
                features.extend(getattr(self, attr))
        _clip(features)
        return features


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
    variables: Optional[Dict[str, Variable]] = None
    constraints: Optional[Dict[str, Constraint]] = None
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
        internal_solver: "InternalSolver",
    ) -> None:
        self.solver = internal_solver

    def extract(
        self,
        instance: "Instance",
        with_static: bool = True,
    ) -> Features:
        features = Features()
        features.variables = self.solver.get_variables_old(
            with_static=with_static,
        )
        features.constraints = self.solver.get_constraints(
            with_static=with_static,
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
        for (var_name, var) in features.variables.items():
            user_features: Optional[List[float]] = None
            category: Category = instance.get_variable_category(var_name)
            if category is not None:
                assert isinstance(category, collections.Hashable), (
                    f"Variable category must be be hashable. "
                    f"Found {type(category).__name__} instead for var={var_name}."
                )
                user_features = instance.get_variable_features(var_name)
                if isinstance(user_features, np.ndarray):
                    user_features = user_features.tolist()
                assert isinstance(user_features, list), (
                    f"Variable features must be a list. "
                    f"Found {type(user_features).__name__} instead for "
                    f"var={var_name}."
                )
                for v in user_features:
                    assert isinstance(v, numbers.Real), (
                        f"Variable features must be a list of numbers. "
                        f"Found {type(v).__name__} instead "
                        f"for var={var_name}."
                    )
            var.category = category
            var.user_features = user_features

    def _extract_user_features_constrs(
        self,
        instance: "Instance",
        features: Features,
    ) -> None:
        assert features.constraints is not None
        has_static_lazy = instance.has_static_lazy_constraints()
        for (cid, constr) in features.constraints.items():
            user_features = None
            category = instance.get_constraint_category(cid)
            if category is not None:
                assert isinstance(category, collections.Hashable), (
                    f"Constraint category must be hashable. "
                    f"Found {type(category).__name__} instead for cid={cid}.",
                )
                user_features = instance.get_constraint_features(cid)
                if isinstance(user_features, np.ndarray):
                    user_features = user_features.tolist()
                assert isinstance(user_features, list), (
                    f"Constraint features must be a list. "
                    f"Found {type(user_features).__name__} instead for cid={cid}."
                )
                assert isinstance(user_features[0], float), (
                    f"Constraint features must be a list of floats. "
                    f"Found {type(user_features[0]).__name__} instead for cid={cid}."
                )
            if has_static_lazy:
                constr.lazy = instance.is_constraint_lazy(cid)
            constr.category = category
            constr.user_features = user_features

    def _extract_user_features_instance(
        self,
        instance: "Instance",
        features: Features,
    ) -> None:
        assert features.constraints is not None
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
        lazy_count = 0
        for (cid, cdict) in features.constraints.items():
            if cdict.lazy:
                lazy_count += 1
        features.instance = InstanceFeatures(
            user_features=user_features,
            lazy_constraint_count=lazy_count,
        )

    def _extract_alvarez_2017(self, features: Features) -> None:
        assert features.variables is not None

        pos_obj_coeff_sum = 0.0
        neg_obj_coeff_sum = 0.0
        for (varname, var) in features.variables.items():
            if var.obj_coeff is not None:
                if var.obj_coeff > 0:
                    pos_obj_coeff_sum += var.obj_coeff
                if var.obj_coeff < 0:
                    neg_obj_coeff_sum += -var.obj_coeff

        for (varname, var) in features.variables.items():
            assert isinstance(var, Variable)
            f: List[float] = []
            if var.obj_coeff is not None:
                # Feature 1
                f.append(np.sign(var.obj_coeff))

                # Feature 2
                if pos_obj_coeff_sum > 0:
                    f.append(abs(var.obj_coeff) / pos_obj_coeff_sum)
                else:
                    f.append(0.0)

                # Feature 3
                if neg_obj_coeff_sum > 0:
                    f.append(abs(var.obj_coeff) / neg_obj_coeff_sum)
                else:
                    f.append(0.0)

            if var.value is not None:
                # Feature 37
                f.append(
                    min(
                        var.value - np.floor(var.value),
                        np.ceil(var.value) - var.value,
                    )
                )

            if var.sa_obj_up is not None:
                assert var.obj_coeff is not None
                assert var.sa_obj_down is not None
                # Convert inf into large finite numbers
                sa_obj_down = max(-1e20, var.sa_obj_down)
                sa_obj_up = min(1e20, var.sa_obj_up)

                # Features 44 and 46
                f.append(np.sign(var.sa_obj_up))
                f.append(np.sign(var.sa_obj_down))

                # Feature 47
                csign = np.sign(var.obj_coeff)
                if csign != 0 and ((var.obj_coeff - sa_obj_down) / csign) > 0.001:
                    f.append(log((var.obj_coeff - sa_obj_down) / csign))
                else:
                    f.append(0.0)

                # Feature 48
                if csign != 0 and ((sa_obj_up - var.obj_coeff) / csign) > 0.001:
                    f.append(log((sa_obj_up - var.obj_coeff) / csign))
                else:
                    f.append(0.0)

            for v in f:
                assert isfinite(v), f"non-finite elements detected: {f}"
            var.alvarez_2017 = f


def _clip(v: List[float]) -> None:
    for (i, vi) in enumerate(v):
        if not isfinite(vi):
            v[i] = max(min(vi, 1e20), -1e20)
