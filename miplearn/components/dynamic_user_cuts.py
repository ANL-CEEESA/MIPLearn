#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
from typing import Any, TYPE_CHECKING, Tuple, Dict, List

import numpy as np
from overrides import overrides

from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.classifiers.threshold import Threshold, MinProbabilityThreshold
from miplearn.components.component import Component
from miplearn.components.dynamic_common import DynamicConstraintsComponent
from miplearn.features.sample import Sample
from miplearn.instance.base import Instance
from miplearn.types import LearningSolveStats, ConstraintName, ConstraintCategory

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
            attr="mip_user_cuts",
        )
        self.enforced: Dict[ConstraintName, Any] = {}
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
        vnames = self.dynamic.sample_predict(instance, sample)
        logger.info("Enforcing %d user cuts ahead-of-time..." % len(vnames))
        for vname in vnames:
            vdata = self.dynamic.known_violations[vname]
            instance.enforce_user_cut(solver.internal_solver, model, vdata)
        stats["UserCuts: Added ahead-of-time"] = len(vnames)

    @overrides
    def user_cut_cb(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
    ) -> None:
        assert solver.internal_solver is not None
        logger.debug("Finding violated user cuts...")
        violations = instance.find_violated_user_cuts(model)
        logger.debug(f"Found {len(violations)} violated user cuts")
        logger.debug("Building violated user cuts...")
        for (vname, vdata) in violations.items():
            if vname in self.enforced:
                continue
            instance.enforce_user_cut(solver.internal_solver, model, vdata)
            self.enforced[vname] = vdata
            self.n_added_in_callback += 1
        if len(violations) > 0:
            logger.debug(f"Added {len(violations)} violated user cuts")

    @overrides
    def after_solve_mip(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        sample.put_scalar("mip_user_cuts", self.dynamic.encode(self.enforced))
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
    ) -> List[ConstraintName]:
        return self.dynamic.sample_predict(instance, sample)

    @overrides
    def pre_sample_xy(self, instance: Instance, sample: Sample) -> Any:
        return self.dynamic.pre_sample_xy(instance, sample)

    @overrides
    def fit_xy(
        self,
        x: Dict[ConstraintCategory, np.ndarray],
        y: Dict[ConstraintCategory, np.ndarray],
    ) -> None:
        self.dynamic.fit_xy(x, y)

    @overrides
    def sample_evaluate(
        self,
        instance: "Instance",
        sample: Sample,
    ) -> Dict[ConstraintCategory, Dict[ConstraintName, float]]:
        return self.dynamic.sample_evaluate(instance, sample)
