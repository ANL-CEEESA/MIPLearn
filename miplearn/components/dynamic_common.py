#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Dict, Hashable, List, Tuple, TYPE_CHECKING

import numpy as np
from overrides import overrides

from miplearn.classifiers import Classifier
from miplearn.classifiers.threshold import Threshold
from miplearn.components import classifier_evaluation_dict
from miplearn.components.component import Component
from miplearn.features import TrainingSample

if TYPE_CHECKING:
    from miplearn.solvers.learning import Instance


class DynamicConstraintsComponent(Component):
    """
    Base component used by both DynamicLazyConstraintsComponent and UserCutsComponent.
    """

    def __init__(
        self,
        attr: str,
        classifier: Classifier,
        threshold: Threshold,
    ):
        assert isinstance(classifier, Classifier)
        self.threshold_prototype: Threshold = threshold
        self.classifier_prototype: Classifier = classifier
        self.classifiers: Dict[Hashable, Classifier] = {}
        self.thresholds: Dict[Hashable, Threshold] = {}
        self.known_cids: List[str] = []
        self.attr = attr

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
            if getattr(sample, self.attr) is not None:
                if cid in getattr(sample, self.attr):
                    y[category] += [[False, True]]
                else:
                    y[category] += [[True, False]]
        return x, y, cids

    @overrides
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
    ) -> List[Hashable]:
        pred: List[Hashable] = []
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

    @overrides
    def fit(self, training_instances: List["Instance"]) -> None:
        collected_cids = set()
        for instance in training_instances:
            instance.load()
            for sample in instance.training_data:
                if getattr(sample, self.attr) is None:
                    continue
                collected_cids |= getattr(sample, self.attr)
            instance.free()
        self.known_cids.clear()
        self.known_cids.extend(sorted(collected_cids))
        super().fit(training_instances)

    @overrides
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

    @overrides
    def sample_evaluate(
        self,
        instance: "Instance",
        sample: TrainingSample,
    ) -> Dict[Hashable, Dict[str, float]]:
        assert getattr(sample, self.attr) is not None
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
                if cid in getattr(sample, self.attr):
                    tp[category] += 1
                else:
                    fp[category] += 1
            else:
                if cid in getattr(sample, self.attr):
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
