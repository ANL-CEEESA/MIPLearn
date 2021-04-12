#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Dict, List, TYPE_CHECKING, Hashable, Tuple, Any, Optional

import numpy as np
from overrides import overrides

from miplearn.instance.base import Instance
from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.classifiers.threshold import MinProbabilityThreshold, Threshold
from miplearn.components.component import Component
from miplearn.components.dynamic_common import DynamicConstraintsComponent
from miplearn.features import TrainingSample, Features, Sample
from miplearn.types import LearningSolveStats

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miplearn.solvers.learning import LearningSolver


class DynamicLazyConstraintsComponent(Component):
    """
    A component that predicts which lazy constraints to enforce.
    """

    def __init__(
        self,
        classifier: Classifier = CountingClassifier(),
        threshold: Threshold = MinProbabilityThreshold([0, 0.05]),
    ):
        self.dynamic: DynamicConstraintsComponent = DynamicConstraintsComponent(
            classifier=classifier,
            threshold=threshold,
            attr="lazy_enforced",
        )
        self.classifiers = self.dynamic.classifiers
        self.thresholds = self.dynamic.thresholds
        self.known_cids = self.dynamic.known_cids

    @staticmethod
    def enforce(
        cids: List[Hashable],
        instance: Instance,
        model: Any,
        solver: "LearningSolver",
    ) -> None:
        assert solver.internal_solver is not None
        for cid in cids:
            instance.enforce_lazy_constraint(solver.internal_solver, model, cid)

    @overrides
    def before_solve_mip_old(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        training_data.lazy_enforced = set()
        logger.info("Predicting violated (dynamic) lazy constraints...")
        cids = self.dynamic.sample_predict(instance, training_data)
        logger.info("Enforcing %d lazy constraints..." % len(cids))
        self.enforce(cids, instance, model, solver)

    @overrides
    def iteration_cb(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
    ) -> bool:
        assert solver.internal_solver is not None
        logger.debug("Finding violated lazy constraints...")
        cids = instance.find_violated_lazy_constraints(solver.internal_solver, model)
        if len(cids) == 0:
            logger.debug("No violations found")
            return False
        else:
            sample = instance.training_data[-1]
            assert sample.lazy_enforced is not None
            sample.lazy_enforced |= set(cids)
            logger.debug("    %d violations found" % len(cids))
            self.enforce(cids, instance, model, solver)
            return True

    # Delegate ML methods to self.dynamic
    # -------------------------------------------------------------------
    @overrides
    def sample_xy_old(
        self,
        instance: Instance,
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
        instance: Instance,
        sample: TrainingSample,
    ) -> List[Hashable]:
        return self.dynamic.sample_predict(instance, sample)

    @overrides
    def fit(self, training_instances: List[Instance]) -> None:
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
        instance: Instance,
        sample: TrainingSample,
    ) -> Dict[Hashable, Dict[str, float]]:
        return self.dynamic.sample_evaluate_old(instance, sample)
