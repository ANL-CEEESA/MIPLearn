#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
from typing import List, Dict, Any, Hashable

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.h5 import H5File
from miplearn.solvers.gurobi import GurobiModel

logger = logging.getLogger(__name__)

# TODO: Replace GurobiModel by AbstractModel
# TODO: fix_violations: remove model.inner
# TODO: fix_violations: remove `where`
# TODO: Write documentation
# TODO: Implement ExpertLazyConstrComponent


class MemorizingLazyConstrComponent:
    def __init__(self, clf: Any, extractor: FeaturesExtractor) -> None:
        self.clf = clf
        self.extractor = extractor
        self.violations_: List[Hashable] = []
        self.n_features_: int = 0
        self.n_targets_: int = 0

    def fit(self, train_h5: List[str]) -> None:
        logger.info("Reading training data...")
        n_samples = len(train_h5)
        x, y, violations, n_features = [], [], [], None
        violation_to_idx: Dict[Hashable, int] = {}
        for h5_filename in train_h5:
            with H5File(h5_filename, "r") as h5:

                # Store lazy constraints
                sample_violations_str = h5.get_scalar("mip_constr_violations")
                assert sample_violations_str is not None
                assert isinstance(sample_violations_str, str)
                sample_violations = eval(sample_violations_str)
                assert isinstance(sample_violations, list)
                y_sample = []
                for v in sample_violations:
                    if v not in violation_to_idx:
                        violation_to_idx[v] = len(violation_to_idx)
                        violations.append(v)
                    y_sample.append(violation_to_idx[v])
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
        self.violations_ = violations
        self.n_targets_ = len(violation_to_idx)
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

    def before_mip(
        self,
        test_h5: str,
        model: GurobiModel,
        stats: Dict[str, Any],
    ) -> None:
        assert self.violations_ is not None
        if model.fix_violations is None:
            return

        # Read features
        with H5File(test_h5, "r") as h5:
            x_sample = self.extractor.get_instance_features(h5)
        assert x_sample.shape == (self.n_features_,)
        x_sample = x_sample.reshape(1, -1)

        # Predict violated constraints
        logger.info("Predicting violated lazy constraints...")
        y = self.clf.predict(x_sample)
        assert y.shape == (1, self.n_targets_)
        y = y.reshape(-1)

        # Enforce constraints
        violations = [self.violations_[i] for (i, yi) in enumerate(y) if yi > 0.5]
        logger.info(f"Enforcing {len(violations)} constraints ahead-of-time...")
        model.fix_violations(model, violations, "aot")
        stats["Lazy Constraints: AOT"] = len(violations)
