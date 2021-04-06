#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Dict, List, TYPE_CHECKING, Hashable, Tuple

import numpy as np

from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.classifiers.threshold import MinProbabilityThreshold, Threshold
from miplearn.components.component import Component
from miplearn.components.dynamic_common import DynamicConstraintsComponent
from miplearn.features import TrainingSample

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miplearn.solvers.learning import Instance


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
    def enforce(cids, instance, model, solver):
        for cid in cids:
            cobj = instance.build_lazy_constraint(model, cid)
            solver.internal_solver.add_constraint(cobj)

    def before_solve_mip(
        self,
        solver,
        instance,
        model,
        stats,
        features,
        training_data,
    ):
        training_data.lazy_enforced = set()
        logger.info("Predicting violated lazy constraints...")
        cids = self.dynamic.sample_predict(instance, training_data)
        logger.info("Enforcing %d lazy constraints..." % len(cids))
        self.enforce(cids, instance, model, solver)

    def iteration_cb(self, solver, instance, model):
        logger.debug("Finding violated lazy constraints...")
        cids = instance.find_violated_lazy_constraints(model)
        if len(cids) == 0:
            logger.debug("No violations found")
            return False
        else:
            instance.training_data[-1].lazy_enforced |= set(cids)
            logger.debug("    %d violations found" % len(cids))
            self.enforce(cids, instance, model, solver)
            return True

    # Delegate ML methods to self.dynamic
    # -------------------------------------------------------------------
    def sample_xy(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> Tuple[Dict, Dict]:
        return self.dynamic.sample_xy(instance, sample)

    def sample_predict(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> List[str]:
        return self.dynamic.sample_predict(instance, sample)

    def fit(self, training_instances: List["Instance"]) -> None:
        self.dynamic.fit(training_instances)

    def fit_xy(
        self,
        x: Dict[Hashable, np.ndarray],
        y: Dict[Hashable, np.ndarray],
    ) -> None:
        self.dynamic.fit_xy(x, y)

    def sample_evaluate(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> Dict[Hashable, Dict[str, float]]:
        return self.dynamic.sample_evaluate(instance, sample)
