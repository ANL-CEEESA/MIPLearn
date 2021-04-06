#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Optional, Any, cast

import numpy as np
import sklearn

from miplearn.classifiers import Classifier, Regressor


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

    def clone(self) -> "ScikitLearnClassifier":
        return ScikitLearnClassifier(
            clf=sklearn.base.clone(self.inner_clf),
        )


class ScikitLearnRegressor(Regressor):
    """
    Wrapper for ScikitLearn regressors, which makes sure inputs and outputs have the
    correct dimensions and types.
    """

    def __init__(self, reg: Any) -> None:
        super().__init__()
        self.inner_reg = reg

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        super().fit(x_train, y_train)
        self.inner_reg.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        super().predict(x_test)
        n_samples = x_test.shape[0]
        sklearn_pred = self.inner_reg.predict(x_test)
        assert isinstance(sklearn_pred, np.ndarray)
        assert sklearn_pred.shape[0] == n_samples
        return sklearn_pred

    def clone(self) -> "ScikitLearnRegressor":
        return ScikitLearnRegressor(
            reg=sklearn.base.clone(self.inner_reg),
        )
