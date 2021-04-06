#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Any, TYPE_CHECKING, Hashable, Set

from miplearn import Component, Instance

import logging

from miplearn.features import Features, TrainingSample
from miplearn.types import LearningSolveStats

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miplearn.solvers.learning import LearningSolver


class UserCutsComponentNG(Component):
    def __init__(self) -> None:
        self.enforced: Set[Hashable] = set()

    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        self.enforced.clear()

    def after_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        training_data.user_cuts_enforced = set(self.enforced)

    def user_cut_cb(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
    ) -> None:
        assert solver.internal_solver is not None
        logger.debug("Finding violated user cuts...")
        cids = instance.find_violated_user_cuts(model)
        logger.debug(f"Found {len(cids)} violated user cuts")
        logger.debug("Building violated user cuts...")
        for cid in cids:
            assert isinstance(cid, Hashable)
            cobj = instance.build_user_cut(model, cid)
            assert cobj is not None
            solver.internal_solver.add_cut(cobj)
            self.enforced.add(cid)
        if len(cids) > 0:
            logger.info(f"Added {len(cids)} violated user cuts")
