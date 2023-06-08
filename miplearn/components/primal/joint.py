#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
from typing import List, Dict, Any, Optional

import numpy as np

from miplearn.components.primal import _extract_bin_var_names_values
from miplearn.components.primal.actions import PrimalComponentAction
from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.solvers.abstract import AbstractModel
from miplearn.h5 import H5File

logger = logging.getLogger(__name__)


class JointVarsPrimalComponent:
    def __init__(
        self, clf: Any, extractor: FeaturesExtractor, action: PrimalComponentAction
    ):
        self.clf = clf
        self.extractor = extractor
        self.bin_var_names_: Optional[np.ndarray] = None
        self.action = action

    def fit(self, train_h5: List[str]) -> None:
        logger.info("Reading training data...")
        self.bin_var_names_ = None
        x, y, n_samples, n_features = [], [], len(train_h5), None
        for h5_filename in train_h5:
            with H5File(h5_filename, "r") as h5:
                bin_var_names, bin_var_values, _ = _extract_bin_var_names_values(h5)

                # Store/check variable names
                if self.bin_var_names_ is None:
                    self.bin_var_names_ = bin_var_names
                else:
                    assert np.all(bin_var_names == self.bin_var_names_)

                # Build x and y vectors
                x_sample = self.extractor.get_instance_features(h5)
                assert len(x_sample.shape) == 1
                if n_features is None:
                    n_features = len(x_sample)
                else:
                    assert len(x_sample) == n_features
                x.append(x_sample)
                y.append(bin_var_values)
        assert self.bin_var_names_ is not None

        logger.info("Constructing matrices...")
        x_np = np.vstack(x)
        y_np = np.array(y)
        assert len(x_np.shape) == 2
        assert x_np.shape[0] == n_samples
        assert x_np.shape[1] == n_features
        assert y_np.shape == (n_samples, len(self.bin_var_names_))
        logger.info(
            f"Dataset has {n_samples:,d} samples, "
            f"{n_features:,d} features and {y_np.shape[1]:,d} targets"
        )

        logger.info("Training classifier...")
        self.clf.fit(x_np, y_np)

        logger.info("Done fitting.")

    def before_mip(
        self, test_h5: str, model: AbstractModel, stats: Dict[str, Any]
    ) -> None:
        assert self.bin_var_names_ is not None

        # Read features
        with H5File(test_h5, "r") as h5:
            x_sample = self.extractor.get_instance_features(h5)
        assert len(x_sample.shape) == 1
        x_sample = x_sample.reshape(1, -1)

        # Predict optimal solution
        logger.info("Predicting warm starts...")
        y_pred = self.clf.predict(x_sample)
        assert len(y_pred.shape) == 2
        assert y_pred.shape[0] == 1
        assert y_pred.shape[1] == len(self.bin_var_names_)

        # Construct warm starts, based on prediction
        self.action.perform(model, self.bin_var_names_, y_pred, stats)
