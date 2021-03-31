#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import TYPE_CHECKING, Dict

from miplearn.types import ModelFeatures, ConstraintFeatures

if TYPE_CHECKING:
    from miplearn import InternalSolver


class ModelFeaturesExtractor:
    def __init__(
        self,
        internal_solver: "InternalSolver",
    ) -> None:
        self.solver = internal_solver

    def extract(self) -> ModelFeatures:
        constraints: Dict[str, ConstraintFeatures] = {}
        for cid in self.solver.get_constraint_ids():
            constraints[cid] = {
                "RHS": self.solver.get_constraint_rhs(cid),
                "LHS": self.solver.get_constraint_lhs(cid),
                "Sense": self.solver.get_constraint_sense(cid),
            }
        return {
            "Constraints": constraints,
            "Variables": self.solver.get_empty_solution(),
        }
