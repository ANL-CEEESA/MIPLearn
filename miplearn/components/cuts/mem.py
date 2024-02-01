#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import json
import logging
from typing import List, Dict, Any, Hashable

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.h5 import H5File
from miplearn.solvers.abstract import AbstractModel

logger = logging.getLogger(__name__)


def convert_lists_to_tuples(obj: Any) -> Any:
    if isinstance(obj, list):
        return tuple(convert_lists_to_tuples(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: convert_lists_to_tuples(value) for key, value in obj.items()}
    else:
        return obj


class _BaseMemorizingConstrComponent:
    def __init__(self, clf: Any, extractor: FeaturesExtractor, field: str) -> None:
        self.clf = clf
        self.extractor = extractor
        self.constrs_: List[Hashable] = []
        self.n_features_: int = 0
        self.n_targets_: int = 0
        self.field = field

    def fit(
        self,
        train_h5: List[str],
    ) -> None:
        logger.info("Reading training data...")
        n_samples = len(train_h5)
        x, y, constrs, n_features = [], [], [], None
        constr_to_idx: Dict[Hashable, int] = {}
        for h5_filename in train_h5:
            with H5File(h5_filename, "r") as h5:
                # Store constraints
                sample_constrs_str = h5.get_scalar(self.field)
                assert sample_constrs_str is not None
                assert isinstance(sample_constrs_str, str)
                sample_constrs = convert_lists_to_tuples(json.loads(sample_constrs_str))
                y_sample = []
                for c in sample_constrs:
                    if c not in constr_to_idx:
                        constr_to_idx[c] = len(constr_to_idx)
                        constrs.append(c)
                    y_sample.append(constr_to_idx[c])
                y.append(y_sample)

                # Extract features
                x_sample = self.extractor.get_instance_features(h5)
                assert len(x_sample.shape) == 1
                if n_features is None:
                    n_features = len(x_sample)
                else:
                    assert len(x_sample) == n_features
                x.append(x_sample)
        logger.info("Constructing matrices...")
        assert n_features is not None
        self.n_features_ = n_features
        self.constrs_ = constrs
        self.n_targets_ = len(constr_to_idx)
        x_np = np.vstack(x)
        assert x_np.shape == (n_samples, n_features)
        y_np = MultiLabelBinarizer().fit_transform(y)
        assert y_np.shape == (n_samples, self.n_targets_)
        logger.info(
            f"Dataset has {n_samples:,d} samples, "
            f"{n_features:,d} features and {self.n_targets_:,d} targets"
        )
        logger.info("Training classifier...")
        self.clf.fit(x_np, y_np)

    def predict(
        self,
        msg: str,
        test_h5: str,
    ) -> List[Hashable]:
        with H5File(test_h5, "r") as h5:
            x_sample = self.extractor.get_instance_features(h5)
        assert x_sample.shape == (self.n_features_,)
        x_sample = x_sample.reshape(1, -1)
        logger.info(msg)
        y = self.clf.predict(x_sample)
        assert y.shape == (1, self.n_targets_)
        y = y.reshape(-1)
        return [self.constrs_[i] for (i, yi) in enumerate(y) if yi > 0.5]


class MemorizingCutsComponent(_BaseMemorizingConstrComponent):
    def __init__(self, clf: Any, extractor: FeaturesExtractor) -> None:
        super().__init__(clf, extractor, "mip_cuts")

    def before_mip(
        self,
        test_h5: str,
        model: AbstractModel,
        stats: Dict[str, Any],
    ) -> None:
        if model.cuts_enforce is None:
            return
        assert self.constrs_ is not None
        model.cuts_aot_ = self.predict("Predicting cutting planes...", test_h5)
        stats["Cuts: AOT"] = len(model.cuts_aot_)
