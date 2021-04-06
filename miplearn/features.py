#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import collections
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Set, List, Hashable

from miplearn.types import VarIndex, Solution

if TYPE_CHECKING:
    from miplearn import InternalSolver, Instance


@dataclass
class TrainingSample:
    lp_log: Optional[str] = None
    lp_solution: Optional[Solution] = None
    lp_value: Optional[float] = None
    lazy_enforced: Optional[Set[str]] = None
    lower_bound: Optional[float] = None
    mip_log: Optional[str] = None
    solution: Optional[Solution] = None
    upper_bound: Optional[float] = None
    slacks: Optional[Dict[str, float]] = None


@dataclass
class InstanceFeatures:
    user_features: Optional[List[float]] = None
    lazy_constraint_count: int = 0


@dataclass
class VariableFeatures:
    category: Optional[Hashable] = None
    user_features: Optional[List[float]] = None


@dataclass
class ConstraintFeatures:
    rhs: Optional[float] = None
    lhs: Optional[Dict[str, float]] = None
    sense: Optional[str] = None
    category: Optional[Hashable] = None
    user_features: Optional[List[float]] = None
    lazy: bool = False


@dataclass
class Features:
    instance: Optional[InstanceFeatures] = None
    variables: Optional[Dict[str, Dict[VarIndex, VariableFeatures]]] = None
    constraints: Optional[Dict[str, ConstraintFeatures]] = None


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
    ) -> Dict[str, Dict[VarIndex, VariableFeatures]]:
        result: Dict[str, Dict[VarIndex, VariableFeatures]] = {}
        empty_solution = self.solver.get_empty_solution()
        for (var_name, var_dict) in empty_solution.items():
            result[var_name] = {}
            for idx in var_dict.keys():
                user_features = None
                category = instance.get_variable_category(var_name, idx)
                if category is not None:
                    assert isinstance(category, collections.Hashable), (
                        f"Variable category must be be hashable. "
                        f"Found {type(category).__name__} instead for var={var_name}."
                    )
                    user_features = instance.get_variable_features(var_name, idx)
                    assert isinstance(user_features, list), (
                        f"Variable features must be a list. "
                        f"Found {type(user_features).__name__} instead for "
                        f"var={var_name}[{idx}]."
                    )
                    for v in user_features:
                        assert isinstance(v, numbers.Real), (
                            f"Variable features must be a list of numbers. "
                            f"Found {type(v).__name__} instead "
                            f"for var={var_name}[{idx}]."
                        )
                result[var_name][idx] = VariableFeatures(
                    category=category,
                    user_features=user_features,
                )
        return result

    def _extract_constraints(
        self,
        instance: "Instance",
    ) -> Dict[str, ConstraintFeatures]:
        has_static_lazy = instance.has_static_lazy_constraints()
        constraints: Dict[str, ConstraintFeatures] = {}
        for cid in self.solver.get_constraint_ids():
            user_features = None
            category = instance.get_constraint_category(cid)
            if category is not None:
                assert isinstance(category, collections.Hashable), (
                    f"Constraint category must be hashable. "
                    f"Found {type(category).__name__} instead for cid={cid}.",
                )
                user_features = instance.get_constraint_features(cid)
                assert isinstance(user_features, list), (
                    f"Constraint features must be a list. "
                    f"Found {type(user_features).__name__} instead for cid={cid}."
                )
                assert isinstance(user_features[0], float), (
                    f"Constraint features must be a list of floats. "
                    f"Found {type(user_features[0]).__name__} instead for cid={cid}."
                )
            constraints[cid] = ConstraintFeatures(
                rhs=self.solver.get_constraint_rhs(cid),
                lhs=self.solver.get_constraint_lhs(cid),
                sense=self.solver.get_constraint_sense(cid),
                category=category,
                user_features=user_features,
            )
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
