#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import ABC, abstractmethod
from typing import Optional, Any, cast

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
        assert x_train.dtype in [np.float16, np.float32, np.float64]
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
        assert n_features_x == self.n_features
        return np.ndarray([])


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
        assert len(x_train.shape) == 2
        assert len(y_train.shape) == 2
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


class ScikitLearnClassifier(Classifier):
    """
    Wrapper for ScikitLearn classifiers, which makes sure inputs and outputs have the
    correct dimensions and types.
    """

    def __init__(self, clf: Any) -> None:
        super().__init__()
        self.inner_clf = clf
        self.constant: Optional[np.ndarray] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        super().fit(x_train, y_train)
        (n_samples, n_classes) = y_train.shape
        assert n_classes == 2, (
            f"Scikit-learn classifiers must have exactly two classes. "
            f"{n_classes} classes were provided instead."
        )

        # When all samples belong to the same class, sklearn's predict_proba returns
        # an array with a single column. The following check avoid this strange
        # behavior.
        mean = cast(np.ndarray, y_train.astype(float).mean(axis=0))
        if mean.max() == 1.0:
            self.constant = mean
            return

        self.inner_clf.fit(x_train, y_train[:, 1])

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        super().predict_proba(x_test)
        n_samples = x_test.shape[0]
        if self.constant is not None:
            return np.array([self.constant for n in range(n_samples)])
        sklearn_proba = self.inner_clf.predict_proba(x_test)
        if isinstance(sklearn_proba, list):
            assert len(sklearn_proba) == self.n_classes
            for pb in sklearn_proba:
                assert isinstance(pb, np.ndarray)
                assert pb.dtype in [np.float16, np.float32, np.float64]
                assert pb.shape == (n_samples, 2)
            proba = np.hstack([pb[:, [1]] for pb in sklearn_proba])
            assert proba.shape == (n_samples, self.n_classes)
            return proba
        else:
            assert isinstance(sklearn_proba, np.ndarray)
            assert sklearn_proba.shape == (n_samples, 2)
            return sklearn_proba
