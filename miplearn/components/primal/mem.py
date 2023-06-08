#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from . import _extract_bin_var_names_values
from .actions import PrimalComponentAction
from ...extractors.abstract import FeaturesExtractor
from ...solvers.abstract import AbstractModel
from ...h5 import H5File

logger = logging.getLogger()


class SolutionConstructor(ABC):
    @abstractmethod
    def construct(self, y_proba: np.ndarray, solutions: np.ndarray) -> np.ndarray:
        pass


class MemorizingPrimalComponent:
    """
    Component that memorizes all solutions seen during training, then fits a
    single classifier to predict which of the memorized solutions should be
    provided to the solver. Optionally combines multiple memorized solutions
    into a single, partial one.
    """

    def __init__(
        self,
        clf: Any,
        extractor: FeaturesExtractor,
        constructor: SolutionConstructor,
        action: PrimalComponentAction,
    ) -> None:
        assert clf is not None
        self.clf = clf
        self.extractor = extractor
        self.constructor = constructor
        self.solutions_: Optional[np.ndarray] = None
        self.bin_var_names_: Optional[np.ndarray] = None
        self.action = action

    def fit(self, train_h5: List[str]) -> None:
        logger.info("Reading training data...")
        n_samples = len(train_h5)
        solutions_ = []
        self.bin_var_names_ = None
        x, y, n_features = [], [], None
        solution_to_idx: Dict[Tuple, int] = {}
        for h5_filename in train_h5:
            with H5File(h5_filename, "r") as h5:
                bin_var_names, bin_var_values, _ = _extract_bin_var_names_values(h5)

                # Store/check variable names
                if self.bin_var_names_ is None:
                    self.bin_var_names_ = bin_var_names
                else:
                    assert np.all(bin_var_names == self.bin_var_names_)

                # Store solution
                sol = tuple(np.where(bin_var_values)[0])
                if sol not in solution_to_idx:
                    solutions_.append(bin_var_values)
                    solution_to_idx[sol] = len(solution_to_idx)
                y.append(solution_to_idx[sol])

                # Extract features
                x_sample = self.extractor.get_instance_features(h5)
                assert len(x_sample.shape) == 1
                if n_features is None:
                    n_features = len(x_sample)
                else:
                    assert len(x_sample) == n_features
                x.append(x_sample)

        logger.info("Constructing matrices...")
        x_np = np.vstack(x)
        y_np = np.array(y)
        assert len(x_np.shape) == 2
        assert x_np.shape[0] == n_samples
        assert x_np.shape[1] == n_features
        assert y_np.shape == (n_samples,)
        self.solutions_ = np.array(solutions_)
        n_classes = len(solution_to_idx)
        logger.info(
            f"Dataset has {n_samples:,d} samples, "
            f"{n_features:,d} features and {n_classes:,d} classes"
        )

        logger.info("Training classifier...")
        self.clf.fit(x_np, y_np)

        logger.info("Done fitting.")

    def before_mip(
        self, test_h5: str, model: AbstractModel, stats: Dict[str, Any]
    ) -> None:
        assert self.solutions_ is not None
        assert self.bin_var_names_ is not None

        # Read features
        with H5File(test_h5, "r") as h5:
            x_sample = self.extractor.get_instance_features(h5)
        assert len(x_sample.shape) == 1
        x_sample = x_sample.reshape(1, -1)

        # Predict optimal solution
        logger.info("Predicting primal solution...")
        y_proba = self.clf.predict_proba(x_sample)
        assert len(y_proba.shape) == 2
        assert y_proba.shape[0] == 1
        assert y_proba.shape[1] == len(self.solutions_)

        # Construct warm starts, based on prediction
        starts = self.constructor.construct(y_proba[0, :], self.solutions_)
        self.action.perform(model, self.bin_var_names_, starts, stats)


class SelectTopSolutions(SolutionConstructor):
    """
    Warm start construction strategy that selects and returns the top k solutions.
    """

    def __init__(self, k: int) -> None:
        self.k = k

    def construct(self, y_proba: np.ndarray, solutions: np.ndarray) -> np.ndarray:
        # Check arguments
        assert len(y_proba.shape) == 1
        assert len(solutions.shape) == 2
        assert len(y_proba) == solutions.shape[0]

        # Select top k solutions
        ind = np.argsort(-y_proba, kind="stable")
        selected = ind[: min(self.k, len(ind))]
        return solutions[selected, :]


class MergeTopSolutions(SolutionConstructor):
    """
    Warm start construction strategy that first selects the top k solutions,
    then merges them into a single solution.

    To merge the solutions, the strategy first computes the mean optimal value of each
    decision variable, then: (i) sets the variable to zero if the mean is below
    thresholds[0]; (ii) sets the variable to one if the mean is above thresholds[1];
    (iii) leaves the variable free otherwise.
    """

    def __init__(self, k: int, thresholds: List[float]):
        assert len(thresholds) == 2
        self.k = k
        self.thresholds = thresholds

    def construct(self, y_proba: np.ndarray, solutions: np.ndarray) -> np.ndarray:
        filtered = SelectTopSolutions(self.k).construct(y_proba, solutions)
        mean = filtered.mean(axis=0)
        start = np.full((1, solutions.shape[1]), float("nan"))
        start[0, mean <= self.thresholds[0]] = 0
        start[0, mean >= self.thresholds[1]] = 1
        return start
