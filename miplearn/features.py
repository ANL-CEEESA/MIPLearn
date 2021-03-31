#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numbers
import collections
from typing import TYPE_CHECKING, Dict

from miplearn.types import Features, ConstraintFeatures

if TYPE_CHECKING:
    from miplearn import InternalSolver, Instance


class FeaturesExtractor:
    def __init__(
        self,
        internal_solver: "InternalSolver",
    ) -> None:
        self.solver = internal_solver

    def extract(self, instance: "Instance") -> Features:
        return {
            "Constraints": self._extract_constraints(instance),
            "Variables": self._extract_variables(instance),
        }

    def _extract_variables(self, instance: "Instance") -> Dict:
        variables = self.solver.get_empty_solution()
        for (var_name, var_dict) in variables.items():
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
                        f"var={var_name}."
                    )
                    assert isinstance(user_features[0], numbers.Real), (
                        f"Variable features must be a list of numbers."
                        f"Found {type(user_features[0]).__name__} instead "
                        f"for var={var_name}."
                    )
                var_dict[idx] = {
                    "Category": category,
                    "User features": user_features,
                }
        return variables

    def _extract_constraints(
        self,
        instance: "Instance",
    ) -> Dict[str, ConstraintFeatures]:
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
            constraints[cid] = {
                "RHS": self.solver.get_constraint_rhs(cid),
                "LHS": self.solver.get_constraint_lhs(cid),
                "Sense": self.solver.get_constraint_sense(cid),
                "Category": category,
                "User features": user_features,
            }
        return constraints
