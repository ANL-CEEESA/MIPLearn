#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
from overrides import overrides

from miplearn.features.extractor import FeaturesExtractor
from miplearn.classifiers import Classifier
from miplearn.classifiers.threshold import Threshold
from miplearn.components import classifier_evaluation_dict
from miplearn.components.component import Component
from miplearn.features.sample import Sample
from miplearn.instance.base import Instance
from miplearn.types import ConstraintCategory, ConstraintName

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
        self.classifiers: Dict[ConstraintCategory, Classifier] = {}
        self.thresholds: Dict[ConstraintCategory, Threshold] = {}
        self.known_cids: List[ConstraintName] = []
        self.attr = attr

    def sample_xy_with_cids(
        self,
        instance: Optional[Instance],
        sample: Sample,
    ) -> Tuple[
        Dict[ConstraintCategory, List[List[float]]],
        Dict[ConstraintCategory, List[List[bool]]],
        Dict[ConstraintCategory, List[ConstraintName]],
    ]:
        if len(self.known_cids) == 0:
            return {}, {}, {}
        assert instance is not None
        x: Dict[ConstraintCategory, List[List[float]]] = {}
        y: Dict[ConstraintCategory, List[List[bool]]] = {}
        cids: Dict[ConstraintCategory, List[ConstraintName]] = {}
        known_cids = np.array(self.known_cids, dtype="S")

        enforced_cids = None
        enforced_cids_np = sample.get_array(self.attr)
        if enforced_cids_np is not None:
            enforced_cids = list(enforced_cids_np)

        # Get user-provided constraint features
        (
            constr_features,
            constr_categories,
            constr_lazy,
        ) = FeaturesExtractor._extract_user_features_constrs(instance, known_cids)

        # Augment with instance features
        instance_features = sample.get_array("static_instance_features")
        assert instance_features is not None
        constr_features = np.hstack(
            [
                instance_features.reshape(1, -1).repeat(len(known_cids), axis=0),
                constr_features,
            ]
        )

        categories = np.unique(constr_categories)
        for c in categories:
            x[c] = constr_features[constr_categories == c].tolist()
            cids[c] = known_cids[constr_categories == c].tolist()
            if enforced_cids is not None:
                tmp = np.isin(cids[c], enforced_cids).reshape(-1, 1)
                y[c] = np.hstack([~tmp, tmp]).tolist()  # type: ignore

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
            known_cids |= set(list(cids))
        self.known_cids.clear()
        self.known_cids.extend(sorted(known_cids))

    def sample_predict(
        self,
        instance: Instance,
        sample: Sample,
    ) -> List[ConstraintName]:
        pred: List[ConstraintName] = []
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
        return sample.get_array(self.attr)

    @overrides
    def fit_xy(
        self,
        x: Dict[ConstraintCategory, np.ndarray],
        y: Dict[ConstraintCategory, np.ndarray],
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
    ) -> Dict[str, float]:
        actual = sample.get_array(self.attr)
        assert actual is not None
        pred = set(self.sample_predict(instance, sample))
        tp, tn, fp, fn = 0, 0, 0, 0
        for cid in self.known_cids:
            if cid in pred:
                if cid in actual:
                    tp += 1
                else:
                    fp += 1
            else:
                if cid in actual:
                    fn += 1
                else:
                    tn += 1
        return classifier_evaluation_dict(tp=tp, tn=tn, fp=fp, fn=fn)
