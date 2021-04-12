#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Any, TYPE_CHECKING, Hashable, Set, Tuple, Dict, List, Optional

import numpy as np
from overrides import overrides

from miplearn.instance.base import Instance
from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.classifiers.threshold import Threshold, MinProbabilityThreshold
from miplearn.components.component import Component
from miplearn.components.dynamic_common import DynamicConstraintsComponent
from miplearn.features import Features, TrainingSample, Sample
from miplearn.types import LearningSolveStats

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miplearn.solvers.learning import LearningSolver


class UserCutsComponent(Component):
    def __init__(
        self,
        classifier: Classifier = CountingClassifier(),
        threshold: Threshold = MinProbabilityThreshold([0.50, 0.50]),
    ) -> None:
        self.dynamic = DynamicConstraintsComponent(
            classifier=classifier,
            threshold=threshold,
            attr="user_cuts_enforced",
        )
        self.enforced: Set[Hashable] = set()
        self.n_added_in_callback = 0

    @overrides
    def before_solve_mip_old(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        assert solver.internal_solver is not None
        self.enforced.clear()
        self.n_added_in_callback = 0
        logger.info("Predicting violated user cuts...")
        cids = self.dynamic.sample_predict(instance, training_data)
        logger.info("Enforcing %d user cuts ahead-of-time..." % len(cids))
        for cid in cids:
            instance.enforce_user_cut(solver.internal_solver, model, cid)
        stats["UserCuts: Added ahead-of-time"] = len(cids)

    @overrides
    def user_cut_cb(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
    ) -> None:
        assert solver.internal_solver is not None
        logger.debug("Finding violated user cuts...")
        cids = instance.find_violated_user_cuts(model)
        logger.debug(f"Found {len(cids)} violated user cuts")
        logger.debug("Building violated user cuts...")
        for cid in cids:
            if cid in self.enforced:
                continue
            assert isinstance(cid, Hashable)
            instance.enforce_user_cut(solver.internal_solver, model, cid)
            self.enforced.add(cid)
            self.n_added_in_callback += 1
        if len(cids) > 0:
            logger.debug(f"Added {len(cids)} violated user cuts")

    @overrides
    def after_solve_mip_old(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        training_data.user_cuts_enforced = set(self.enforced)
        stats["UserCuts: Added in callback"] = self.n_added_in_callback
        if self.n_added_in_callback > 0:
            logger.info(f"{self.n_added_in_callback} user cuts added in callback")

    # Delegate ML methods to self.dynamic
    # -------------------------------------------------------------------
    @overrides
    def sample_xy_old(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> Tuple[Dict, Dict]:
        return self.dynamic.sample_xy_old(instance, sample)

    @overrides
    def sample_xy(
        self,
        instance: Optional[Instance],
        sample: Sample,
    ) -> Tuple[Dict, Dict]:
        return self.dynamic.sample_xy(instance, sample)

    def sample_predict(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> List[Hashable]:
        return self.dynamic.sample_predict(instance, sample)

    @overrides
    def fit(self, training_instances: List["Instance"]) -> None:
        self.dynamic.fit(training_instances)

    @overrides
    def fit_xy(
        self,
        x: Dict[Hashable, np.ndarray],
        y: Dict[Hashable, np.ndarray],
    ) -> None:
        self.dynamic.fit_xy(x, y)

    @overrides
    def sample_evaluate_old(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> Dict[Hashable, Dict[str, float]]:
        return self.dynamic.sample_evaluate_old(instance, sample)
