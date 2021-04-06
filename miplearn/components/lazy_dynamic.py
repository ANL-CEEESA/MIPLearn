#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Dict, List, TYPE_CHECKING, Hashable, Tuple

import numpy as np

from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.classifiers.threshold import MinProbabilityThreshold, Threshold
from miplearn.components import classifier_evaluation_dict
from miplearn.components.component import Component
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
        assert isinstance(classifier, Classifier)
        self.threshold_prototype: Threshold = threshold
        self.classifier_prototype: Classifier = classifier
        self.classifiers: Dict[Hashable, Classifier] = {}
        self.thresholds: Dict[Hashable, Threshold] = {}
        self.known_cids: List[str] = []

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
        cids = self.sample_predict(instance, training_data)
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

    def sample_xy_with_cids(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> Tuple[
        Dict[Hashable, List[List[float]]],
        Dict[Hashable, List[List[bool]]],
        Dict[Hashable, List[str]],
    ]:
        x: Dict[Hashable, List[List[float]]] = {}
        y: Dict[Hashable, List[List[bool]]] = {}
        cids: Dict[Hashable, List[str]] = {}
        for cid in self.known_cids:
            category = instance.get_constraint_category(cid)
            if category is None:
                continue
            if category not in x:
                x[category] = []
                y[category] = []
                cids[category] = []
            assert instance.features.instance is not None
            assert instance.features.instance.user_features is not None
            cfeatures = instance.get_constraint_features(cid)
            assert cfeatures is not None
            assert isinstance(cfeatures, list)
            for ci in cfeatures:
                assert isinstance(ci, float)
            f = list(instance.features.instance.user_features)
            f += cfeatures
            x[category] += [f]
            cids[category] += [cid]
            if sample.lazy_enforced is not None:
                if cid in sample.lazy_enforced:
                    y[category] += [[False, True]]
                else:
                    y[category] += [[True, False]]
        return x, y, cids

    def sample_xy(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> Tuple[Dict, Dict]:
        x, y, _ = self.sample_xy_with_cids(instance, sample)
        return x, y

    def sample_predict(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> List[str]:
        pred: List[str] = []
        x, _, cids = self.sample_xy_with_cids(instance, sample)
        for category in x.keys():
            assert category in self.classifiers
            assert category in self.thresholds
            clf = self.classifiers[category]
            thr = self.thresholds[category]
            nx = np.array(x[category])
            proba = clf.predict_proba(nx)
            t = thr.predict(nx)
            for i in range(proba.shape[0]):
                if proba[i][1] > t[1]:
                    pred += [cids[category][i]]
        return pred

    def fit(self, training_instances: List["Instance"]) -> None:
        self.known_cids.clear()
        for instance in training_instances:
            for sample in instance.training_data:
                if sample.lazy_enforced is None:
                    continue
                self.known_cids += list(sample.lazy_enforced)
        self.known_cids = sorted(set(self.known_cids))
        super().fit(training_instances)

    def fit_xy(
        self,
        x: Dict[Hashable, np.ndarray],
        y: Dict[Hashable, np.ndarray],
    ) -> None:
        for category in x.keys():
            self.classifiers[category] = self.classifier_prototype.clone()
            self.thresholds[category] = self.threshold_prototype.clone()
            npx = np.array(x[category])
            npy = np.array(y[category])
            self.classifiers[category].fit(npx, npy)
            self.thresholds[category].fit(self.classifiers[category], npx, npy)

    def sample_evaluate(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> Dict[Hashable, Dict[str, float]]:
        assert sample.lazy_enforced is not None
        pred = set(self.sample_predict(instance, sample))
        tp: Dict[Hashable, int] = {}
        tn: Dict[Hashable, int] = {}
        fp: Dict[Hashable, int] = {}
        fn: Dict[Hashable, int] = {}
        for cid in self.known_cids:
            category = instance.get_constraint_category(cid)
            if category is None:
                continue
            if category not in tp.keys():
                tp[category] = 0
                tn[category] = 0
                fp[category] = 0
                fn[category] = 0
            if cid in pred:
                if cid in sample.lazy_enforced:
                    tp[category] += 1
                else:
                    fp[category] += 1
            else:
                if cid in sample.lazy_enforced:
                    fn[category] += 1
                else:
                    tn[category] += 1
        return {
            category: classifier_evaluation_dict(
                tp=tp[category],
                tn=tn[category],
                fp=fp[category],
                fn=fn[category],
            )
            for category in tp.keys()
        }
