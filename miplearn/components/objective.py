#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import List, Dict, Union, Optional, Any, TYPE_CHECKING, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    r2_score,
)

from miplearn.classifiers import Regressor
from miplearn.classifiers.sklearn import ScikitLearnRegressor
from miplearn.components.component import Component
from miplearn.extractors import InstanceIterator
from miplearn.instance import Instance
from miplearn.types import TrainingSample, LearningSolveStats, Features

if TYPE_CHECKING:
    from miplearn.solvers.learning import LearningSolver

logger = logging.getLogger(__name__)


class ObjectiveValueComponent(Component):
    """
    A Component which predicts the optimal objective value of the problem.
    """

    def __init__(
        self,
        lb_regressor: Regressor = ScikitLearnRegressor(LinearRegression()),
        ub_regressor: Regressor = ScikitLearnRegressor(LinearRegression()),
    ) -> None:
        assert isinstance(lb_regressor, Regressor)
        assert isinstance(ub_regressor, Regressor)
        self.ub_regressor: Optional[Regressor] = None
        self.lb_regressor: Optional[Regressor] = None
        self.lb_regressor_prototype = lb_regressor
        self.ub_regressor_prototype = ub_regressor
        self._predicted_ub: Optional[float] = None
        self._predicted_lb: Optional[float] = None

    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
    ) -> None:
        if self.ub_regressor is not None:
            logger.info("Predicting optimal value...")
            pred = self.predict([instance])
            self._predicted_lb = pred["Upper bound"][0]
            self._predicted_ub = pred["Lower bound"][0]
            logger.info(
                "Predicted values: lb=%.2f, ub=%.2f"
                % (
                    self._predicted_lb,
                    self._predicted_ub,
                )
            )

    def after_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        training_data: TrainingSample,
    ) -> None:
        if self._predicted_ub is not None:
            stats["Objective: predicted UB"] = self._predicted_ub
        if self._predicted_lb is not None:
            stats["Objective: predicted LB"] = self._predicted_lb

    def fit(self, training_instances: Union[List[str], List[Instance]]) -> None:
        self.lb_regressor = self.lb_regressor_prototype.clone()
        self.ub_regressor = self.ub_regressor_prototype.clone()
        logger.debug("Extracting features...")
        x_train = self.x(training_instances)
        y_train = self.y(training_instances)
        logger.debug("Fitting lb_regressor...")
        self.lb_regressor.fit(x_train, y_train["Lower bound"])
        logger.debug("Fitting ub_regressor...")
        self.ub_regressor.fit(x_train, y_train["Upper bound"])

    def predict(
        self,
        instances: Union[List[str], List[Instance]],
    ) -> Dict[str, List[float]]:
        assert self.lb_regressor is not None
        assert self.ub_regressor is not None
        x_test = self.x(instances)
        (n_samples, n_features) = x_test.shape
        lb = self.lb_regressor.predict(x_test)
        ub = self.ub_regressor.predict(x_test)
        assert lb.shape == (n_samples, 1)
        assert ub.shape == (n_samples, 1)
        return {
            "Lower bound": lb.ravel().tolist(),
            "Upper bound": ub.ravel().tolist(),
        }

    @staticmethod
    def x(instances: Union[List[str], List[Instance]]) -> np.ndarray:
        result = []
        for instance in InstanceIterator(instances):
            for sample in instance.training_data:
                result.append(instance.get_instance_features() + [sample["LP value"]])
        return np.array(result)

    @staticmethod
    def y(instances: Union[List[str], List[Instance]]) -> Dict[str, np.ndarray]:
        ub: List[List[float]] = []
        lb: List[List[float]] = []
        for instance in InstanceIterator(instances):
            for sample in instance.training_data:
                lb.append([sample["Lower bound"]])
                ub.append([sample["Upper bound"]])
        return {
            "Lower bound": np.array(lb),
            "Upper bound": np.array(ub),
        }

    def evaluate(
        self,
        instances: Union[List[str], List[Instance]],
    ) -> Dict[str, Dict[str, float]]:
        y_pred = self.predict(instances)
        y_true = np.array(
            [
                [
                    inst.training_data[0]["Lower bound"],
                    inst.training_data[0]["Upper bound"],
                ]
                for inst in InstanceIterator(instances)
            ]
        )
        y_pred_lb = y_pred["Lower bound"]
        y_pred_ub = y_pred["Upper bound"]
        y_true_lb, y_true_ub = y_true[:, 1], y_true[:, 1]
        ev = {
            "Lower bound": {
                "Mean squared error": mean_squared_error(y_true_lb, y_pred_lb),
                "Explained variance": explained_variance_score(y_true_lb, y_pred_lb),
                "Max error": max_error(y_true_lb, y_pred_lb),
                "Mean absolute error": mean_absolute_error(y_true_lb, y_pred_lb),
                "R2": r2_score(y_true_lb, y_pred_lb),
                "Median absolute error": mean_absolute_error(y_true_lb, y_pred_lb),
            },
            "Upper bound": {
                "Mean squared error": mean_squared_error(y_true_ub, y_pred_ub),
                "Explained variance": explained_variance_score(y_true_ub, y_pred_ub),
                "Max error": max_error(y_true_ub, y_pred_ub),
                "Mean absolute error": mean_absolute_error(y_true_ub, y_pred_ub),
                "R2": r2_score(y_true_ub, y_pred_ub),
                "Median absolute error": mean_absolute_error(y_true_ub, y_pred_ub),
            },
        }
        return ev

    @staticmethod
    def xy(
        features: Features,
        sample: TrainingSample,
    ) -> Tuple[Dict, Dict]:
        f = features["Instance"]["User features"]
        if "LP value" in sample and sample["LP value"] is not None:
            f += [sample["LP value"]]
        x = {
            "Lower bound": [f],
            "Upper bound": [f],
        }
        if "Lower bound" in sample:
            y = {
                "Lower bound": [[sample["Lower bound"]]],
                "Upper bound": [[sample["Upper bound"]]],
            }
            return x, y
        else:
            return x, {}
