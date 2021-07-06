#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Dict, Hashable, List, Tuple, Optional, Any, Set

import numpy as np
from overrides import overrides

from miplearn.classifiers import Classifier
from miplearn.classifiers.threshold import Threshold
from miplearn.components import classifier_evaluation_dict
from miplearn.components.component import Component
from miplearn.features import Sample
from miplearn.instance.base import Instance

logger = logging.getLogger(__name__)


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
        instance: Optional[Instance],
        sample: Sample,
    ) -> Tuple[
        Dict[Hashable, List[List[float]]],
        Dict[Hashable, List[List[bool]]],
        Dict[Hashable, List[str]],
    ]:
        assert instance is not None
        x: Dict[Hashable, List[List[float]]] = {}
        y: Dict[Hashable, List[List[bool]]] = {}
        cids: Dict[Hashable, List[str]] = {}
        constr_categories_dict = instance.get_constraint_categories()
        constr_features_dict = instance.get_constraint_features()
        instance_features = sample.get("instance_features_user")
        assert instance_features is not None
        for cid in self.known_cids:
            # Initialize categories
            if cid in constr_categories_dict:
                category = constr_categories_dict[cid]
            else:
                category = cid
            if category is None:
                continue
            if category not in x:
                x[category] = []
                y[category] = []
                cids[category] = []

            # Features
            features: List[float] = []
            features.extend(instance_features)
            if cid in constr_features_dict:
                features.extend(constr_features_dict[cid])
            for ci in features:
                assert isinstance(ci, float), (
                    f"Constraint features must be a list of floats. "
                    f"Found {ci.__class__.__name__} instead."
                )
            x[category].append(features)
            cids[category].append(cid)

            # Labels
            enforced_cids = sample.get(self.attr)
            if enforced_cids is not None:
                if cid in enforced_cids:
                    y[category] += [[False, True]]
                else:
                    y[category] += [[True, False]]
        return x, y, cids

    @overrides
    def sample_xy(
        self,
        instance: Optional[Instance],
        sample: Sample,
    ) -> Tuple[Dict, Dict]:
        x, y, _ = self.sample_xy_with_cids(instance, sample)
        return x, y

    @overrides
    def pre_fit(self, pre: List[Any]) -> None:
        assert pre is not None
        known_cids: Set = set()
        for cids in pre:
            known_cids |= cids
        self.known_cids.clear()
        self.known_cids.extend(sorted(known_cids))

    def sample_predict(
        self,
        instance: Instance,
        sample: Sample,
    ) -> List[Hashable]:
        pred: List[Hashable] = []
        if len(self.known_cids) == 0:
            logger.info("Classifiers not fitted. Skipping.")
            return pred
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
    def pre_sample_xy(self, instance: Instance, sample: Sample) -> Any:
        return sample.get(self.attr)

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
        instance: Instance,
        sample: Sample,
    ) -> Dict[Hashable, Dict[str, float]]:
        actual = sample.get(self.attr)
        assert actual is not None
        pred = set(self.sample_predict(instance, sample))
        tp: Dict[Hashable, int] = {}
        tn: Dict[Hashable, int] = {}
        fp: Dict[Hashable, int] = {}
        fn: Dict[Hashable, int] = {}
        constr_categories_dict = instance.get_constraint_categories()
        for cid in self.known_cids:
            if cid not in constr_categories_dict:
                continue
            category = constr_categories_dict[cid]
            if category not in tp.keys():
                tp[category] = 0
                tn[category] = 0
                fp[category] = 0
                fn[category] = 0
            if cid in pred:
                if cid in actual:
                    tp[category] += 1
                else:
                    fp[category] += 1
            else:
                if cid in actual:
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
