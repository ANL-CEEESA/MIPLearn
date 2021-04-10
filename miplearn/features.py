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
class VariableFeatures:
    category: Optional[Hashable] = None
    user_features: Optional[List[float]] = None


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
    variables: Optional[Dict[str, VariableFeatures]] = None
    constraints: Optional[Dict[str, Constraint]] = None


class FeaturesExtractor:
    def __init__(
        self,
        internal_solver: "InternalSolver",
    ) -> None:
        self.solver = internal_solver

    def extract(self, instance: "Instance") -> None:
        instance.features.variables = self._extract_variables(instance)
        instance.features.constraints = self._extract_constraints(instance)
        instance.features.instance = self._extract_instance(instance, instance.features)

    def _extract_variables(
        self,
        instance: "Instance",
    ) -> Dict[VariableName, VariableFeatures]:
        result: Dict[VariableName, VariableFeatures] = {}
        for var_name in self.solver.get_variable_names():
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
            result[var_name] = VariableFeatures(
                category=category,
                user_features=user_features,
            )
        return result

    def _extract_constraints(
        self,
        instance: "Instance",
    ) -> Dict[str, Constraint]:
        has_static_lazy = instance.has_static_lazy_constraints()
        constraints = self.solver.get_constraints()

        for (cid, constr) in constraints.items():
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
            constraints[cid].category = category
            constraints[cid].user_features = user_features
            if has_static_lazy:
                constraints[cid].lazy = instance.is_constraint_lazy(cid)
        return constraints

    @staticmethod
    def _extract_instance(
        instance: "Instance",
        features: Features,
    ) -> InstanceFeatures:
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
        return InstanceFeatures(
            user_features=user_features,
            lazy_constraint_count=lazy_count,
        )
