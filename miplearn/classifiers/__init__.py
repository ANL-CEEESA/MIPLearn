#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Classifier(ABC):
    """
    A Classifier decides which class each sample belongs to, based on historical
    data.
    """

    def __init__(self) -> None:
        self.n_features: Optional[int] = None
        self.n_classes: Optional[int] = None

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the classifier.

        Parameters
        ----------
        x_train: np.ndarray
            An array of features with shape (`n_samples`, `n_features`). Each entry
            must be a float.
        y_train: np.ndarray
            An array of labels with shape (`n_samples`, `n_classes`). Each entry must be
            a bool, and there must be exactly one True element in each row.
        """
        assert isinstance(x_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert x_train.dtype in [
            np.float16,
            np.float32,
            np.float64,
        ], f"x_train.dtype shoule be float. Found {x_train.dtype} instead."
        assert y_train.dtype == np.bool8
        assert len(x_train.shape) == 2
        assert len(y_train.shape) == 2
        (n_samples_x, n_features) = x_train.shape
        (n_samples_y, n_classes) = y_train.shape
        assert n_samples_y == n_samples_x
        self.n_features = n_features
        self.n_classes = n_classes

    @abstractmethod
    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of each sample belonging to each class. Must be called
        after fit.

        Parameters
        ----------
        x_test: np.ndarray
            An array of features with shape (`n_samples`, `n_features`). The number of
            features in `x_test` must match the number of features in `x_train` provided
            to `fit`.

        Returns
        -------
        np.ndarray
            An array of predicted probabilities with shape (`n_samples`, `n_classes`),
            where `n_classes` is the number of columns in `y_train` provided to `fit`.
        """
        assert self.n_features is not None
        assert isinstance(x_test, np.ndarray)
        assert len(x_test.shape) == 2
        (n_samples, n_features_x) = x_test.shape
        assert n_features_x == self.n_features, (
            f"Test and training data have different number of "
            f"features: {n_features_x} != {self.n_features}"
        )
        return np.ndarray([])

    @abstractmethod
    def clone(self) -> "Classifier":
        """
        Returns an unfitted copy of this classifier with the same hyperparameters.
        """
        pass


class Regressor(ABC):
    """
    A Regressor tries to predict the values of some continous variables, given the
    values of other variables.
    """

    def __init__(self) -> None:
        self.n_inputs: Optional[int] = None

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the regressor.

        Parameters
        ----------
        x_train: np.ndarray
            An array of inputs with shape (`n_samples`, `n_inputs`). Each entry must be
            a float.
        y_train: np.ndarray
            An array of outputs with shape (`n_samples`, `n_outputs`). Each entry must
            be a float.
        """
        assert isinstance(x_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert x_train.dtype in [np.float16, np.float32, np.float64]
        assert y_train.dtype in [np.float16, np.float32, np.float64]
        assert len(x_train.shape) == 2, (
            f"Parameter x_train should be a square matrix. "
            f"Found {x_train.shape} ndarray instead."
        )
        assert len(y_train.shape) == 2, (
            f"Parameter y_train should be a square matrix. "
            f"Found {y_train.shape} ndarray instead."
        )
        (n_samples_x, n_inputs) = x_train.shape
        (n_samples_y, n_outputs) = y_train.shape
        assert n_samples_y == n_samples_x
        self.n_inputs = n_inputs

    @abstractmethod
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predicts the values of the output variables. Must be called after fit.

        Parameters
        ----------
        x_test: np.ndarray
            An array of inputs with shape (`n_samples`, `n_inputs`), where `n_inputs`
            must match the number of columns in `x_train` provided to `fit`.

        Returns
        -------
        np.ndarray
            An array of outputs  with shape (`n_samples`, `n_outputs`), where
            `n_outputs` is the number of columns in `y_train` provided to `fit`.
        """
        assert self.n_inputs is not None
        assert isinstance(x_test, np.ndarray)
        assert len(x_test.shape) == 2
        (n_samples, n_inputs_x) = x_test.shape
        assert n_inputs_x == self.n_inputs
        return np.ndarray([])

    @abstractmethod
    def clone(self) -> "Regressor":
        """
        Returns an unfitted copy of this regressor with the same hyperparameters.
        """
        pass
