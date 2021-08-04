#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Any, TYPE_CHECKING, Set, Tuple, Dict, List, Optional

import numpy as np
from overrides import overrides

from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.classifiers.threshold import Threshold, MinProbabilityThreshold
from miplearn.components.component import Component
from miplearn.components.dynamic_common import DynamicConstraintsComponent
from miplearn.features.sample import Sample
from miplearn.instance.base import Instance
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
            attr="mip_user_cuts_enforced",
        )
        self.enforced: Set[str] = set()
        self.n_added_in_callback = 0

    @overrides
    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        assert solver.internal_solver is not None
        self.enforced.clear()
        self.n_added_in_callback = 0
        logger.info("Predicting violated user cuts...")
        cids = self.dynamic.sample_predict(instance, sample)
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
            assert isinstance(cid, str)
            instance.enforce_user_cut(solver.internal_solver, model, cid)
            self.enforced.add(cid)
            self.n_added_in_callback += 1
        if len(cids) > 0:
            logger.debug(f"Added {len(cids)} violated user cuts")

    @overrides
    def after_solve_mip(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        sample.put_set("mip_user_cuts_enforced", set(self.enforced))
        stats["UserCuts: Added in callback"] = self.n_added_in_callback
        if self.n_added_in_callback > 0:
            logger.info(f"{self.n_added_in_callback} user cuts added in callback")

    # Delegate ML methods to self.dynamic
    # -------------------------------------------------------------------
    @overrides
    def sample_xy(
        self,
        instance: "Instance",
        sample: Sample,
    ) -> Tuple[Dict, Dict]:
        return self.dynamic.sample_xy(instance, sample)

    @overrides
    def pre_fit(self, pre: List[Any]) -> None:
        self.dynamic.pre_fit(pre)

    def sample_predict(
        self,
        instance: "Instance",
        sample: Sample,
    ) -> List[str]:
        return self.dynamic.sample_predict(instance, sample)

    @overrides
    def pre_sample_xy(self, instance: Instance, sample: Sample) -> Any:
        return self.dynamic.pre_sample_xy(instance, sample)

    @overrides
    def fit_xy(
        self,
        x: Dict[str, np.ndarray],
        y: Dict[str, np.ndarray],
    ) -> None:
        self.dynamic.fit_xy(x, y)

    @overrides
    def sample_evaluate(
        self,
        instance: "Instance",
        sample: Sample,
    ) -> Dict[str, Dict[str, float]]:
        return self.dynamic.sample_evaluate(instance, sample)
