#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
from typing import Any, Dict, List, Callable, Optional

import numpy as np
import sklearn

from miplearn.components.primal import (
    _extract_bin_var_names_values,
    _extract_bin_var_names,
)
from miplearn.components.primal.actions import PrimalComponentAction
from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.solvers.abstract import AbstractModel
from miplearn.h5 import H5File

logger = logging.getLogger(__name__)


class IndependentVarsPrimalComponent:
    def __init__(
        self,
        base_clf: Any,
        extractor: FeaturesExtractor,
        action: PrimalComponentAction,
        clone_fn: Callable[[Any], Any] = sklearn.clone,
    ):
        self.base_clf = base_clf
        self.extractor = extractor
        self.clf_: Dict[bytes, Any] = {}
        self.bin_var_names_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.clone_fn = clone_fn
        self.action = action

    def fit(self, train_h5: List[str]) -> None:
        logger.info("Reading training data...")
        self.bin_var_names_ = None
        n_bin_vars: Optional[int] = None
        n_vars: Optional[int] = None
        x, y = [], []
        for h5_filename in train_h5:
            with H5File(h5_filename, "r") as h5:
                # Get number of variables
                var_types = h5.get_array("static_var_types")
                assert var_types is not None
                n_vars = len(var_types)

                # Extract features
                (
                    bin_var_names,
                    bin_var_values,
                    bin_var_indices,
                ) = _extract_bin_var_names_values(h5)

                # Store/check variable names
                if self.bin_var_names_ is None:
                    self.bin_var_names_ = bin_var_names
                    n_bin_vars = len(self.bin_var_names_)
                else:
                    assert np.all(bin_var_names == self.bin_var_names_)

                # Build x and y vectors
                x_sample = self.extractor.get_var_features(h5)
                assert len(x_sample.shape) == 2
                assert x_sample.shape[0] == n_vars
                x_sample = x_sample[bin_var_indices]
                if self.n_features_ is None:
                    self.n_features_ = x_sample.shape[1]
                else:
                    assert x_sample.shape[1] == self.n_features_
                x.append(x_sample)
                y.append(bin_var_values)

        assert n_bin_vars is not None
        assert self.bin_var_names_ is not None

        logger.info("Constructing matrices...")
        x_np = np.vstack(x)
        y_np = np.hstack(y)
        n_samples = len(train_h5) * n_bin_vars
        assert x_np.shape == (n_samples, self.n_features_)
        assert y_np.shape == (n_samples,)
        logger.info(
            f"Dataset has {n_bin_vars} binary variables, "
            f"{len(train_h5):,d} samples per variable, "
            f"{self.n_features_:,d} features, 1 target and 2 classes"
        )

        logger.info(f"Training {n_bin_vars} classifiers...")
        self.clf_ = {}
        for var_idx, var_name in enumerate(self.bin_var_names_):
            self.clf_[var_name] = self.clone_fn(self.base_clf)
            self.clf_[var_name].fit(
                x_np[var_idx::n_bin_vars, :], y_np[var_idx::n_bin_vars]
            )

        logger.info("Done fitting.")

    def before_mip(
        self, test_h5: str, model: AbstractModel, stats: Dict[str, Any]
    ) -> None:
        assert self.bin_var_names_ is not None
        assert self.n_features_ is not None

        # Read features
        with H5File(test_h5, "r") as h5:
            x_sample = self.extractor.get_var_features(h5)
            bin_var_names, bin_var_indices = _extract_bin_var_names(h5)
            assert np.all(bin_var_names == self.bin_var_names_)
            x_sample = x_sample[bin_var_indices]

        assert x_sample.shape == (len(self.bin_var_names_), self.n_features_)

        # Predict optimal solution
        logger.info("Predicting warm starts...")
        y_pred = []
        for var_idx, var_name in enumerate(self.bin_var_names_):
            x_var = x_sample[var_idx, :].reshape(1, -1)
            y_var = self.clf_[var_name].predict(x_var)
            assert y_var.shape == (1,)
            y_pred.append(y_var[0])

        # Construct warm starts, based on prediction
        y_pred_np = np.array(y_pred).reshape(1, -1)
        assert y_pred_np.shape == (1, len(self.bin_var_names_))
        self.action.perform(model, self.bin_var_names_, y_pred_np, stats)
