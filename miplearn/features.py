#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import collections
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Set, List, Hashable

from miplearn.types import Solution, VariableName, Category
import numpy as np

if TYPE_CHECKING:
    from miplearn.solvers.internal import InternalSolver
    from miplearn.instance.base import Instance


@dataclass
class TrainingSample:
    lp_log: Optional[str] = None
    lp_solution: Optional[Solution] = None
    lp_value: Optional[float] = None
    lazy_enforced: Optional[Set[Hashable]] = None
    lower_bound: Optional[float] = None
    mip_log: Optional[str] = None
    solution: Optional[Solution] = None
    upper_bound: Optional[float] = None
    slacks: Optional[Dict[str, float]] = None
    user_cuts_enforced: Optional[Set[Hashable]] = None


@dataclass
class InstanceFeatures:
    user_features: Optional[List[float]] = None
    lazy_constraint_count: int = 0


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


@dataclass
class Constraint:
    basis_status: Optional[str] = None
    category: Optional[Hashable] = None
    dual_value: Optional[float] = None
    lazy: bool = False
    lhs: Dict[str, float] = lambda: {}  # type: ignore
    rhs: float = 0.0
    sa_rhs_down: Optional[float] = None
    sa_rhs_up: Optional[float] = None
    sense: str = "<"
    slack: Optional[float] = None
    user_features: Optional[List[float]] = None


@dataclass
class Features:
    instance: Optional[InstanceFeatures] = None
    variables: Optional[Dict[str, Variable]] = None
    constraints: Optional[Dict[str, Constraint]] = None


class FeaturesExtractor:
    def __init__(
        self,
        internal_solver: "InternalSolver",
    ) -> None:
        self.solver = internal_solver

    def extract(self, instance: "Instance") -> None:
        instance.features.variables = self.solver.get_variables()
        instance.features.constraints = self.solver.get_constraints()
        self._extract_user_features_vars(instance)
        self._extract_user_features_constrs(instance)
        self._extract_user_features_instance(instance)

    def _extract_user_features_vars(self, instance: "Instance"):
        for (var_name, var) in instance.features.variables.items():
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

    def _extract_user_features_constrs(self, instance: "Instance"):
        has_static_lazy = instance.has_static_lazy_constraints()
        for (cid, constr) in instance.features.constraints.items():
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

    def _extract_user_features_instance(self, instance: "Instance"):
        assert instance.features.constraints is not None
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
        for (cid, cdict) in instance.features.constraints.items():
            if cdict.lazy:
                lazy_count += 1
        instance.features.instance = InstanceFeatures(
            user_features=user_features,
            lazy_constraint_count=lazy_count,
        )
