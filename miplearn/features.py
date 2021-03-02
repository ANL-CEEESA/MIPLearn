#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import TYPE_CHECKING

from miplearn.types import ModelFeatures

if TYPE_CHECKING:
    from miplearn import InternalSolver


class ModelFeaturesExtractor:
    def __init__(
        self,
        internal_solver: "InternalSolver",
    ) -> None:
        self.internal_solver = internal_solver

    def extract(self) -> ModelFeatures:
        rhs = {}
        for cid in self.internal_solver.get_constraint_ids():
            rhs[cid] = self.internal_solver.get_constraint_rhs(cid)
        return {
            "ConstraintRHS": rhs,
        }
