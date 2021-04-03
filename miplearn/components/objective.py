#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import List, Dict, Union, Optional, Any, TYPE_CHECKING, Tuple, Hashable

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
        regressor: Regressor = ScikitLearnRegressor(LinearRegression()),
    ) -> None:
        assert isinstance(regressor, Regressor)
        self.ub_regressor: Optional[Regressor] = None
        self.lb_regressor: Optional[Regressor] = None
        self.regressor_prototype = regressor
        self._predicted_ub: Optional[float] = None
        self._predicted_lb: Optional[float] = None

    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        logger.info("Predicting optimal value...")
        pred = self.sample_predict(features, training_data)
        if "Upper bound" in pred:
            ub = pred["Upper bound"]
            logger.info("Predicted upper bound: %.6e" % ub)
            stats["Objective: Predicted UB"] = ub
        if "Lower bound" in pred:
            lb = pred["Lower bound"]
            logger.info("Predicted lower bound: %.6e" % lb)
            stats["Objective: Predicted LB"] = lb

    def fit_xy(
        self,
        x: Dict[str, np.ndarray],
        y: Dict[str, np.ndarray],
    ) -> None:
        if "Lower bound" in y:
            self.lb_regressor = self.regressor_prototype.clone()
            self.lb_regressor.fit(x["Lower bound"], y["Lower bound"])
        if "Upper bound" in y:
            self.ub_regressor = self.regressor_prototype.clone()
            self.ub_regressor.fit(x["Upper bound"], y["Upper bound"])

    # def evaluate(
    #     self,
    #     instances: Union[List[str], List[Instance]],
    # ) -> Dict[str, Dict[str, float]]:
    #     y_pred = self.predict(instances)
    #     y_true = np.array(
    #         [
    #             [
    #                 inst.training_data[0]["Lower bound"],
    #                 inst.training_data[0]["Upper bound"],
    #             ]
    #             for inst in InstanceIterator(instances)
    #         ]
    #     )
    #     y_pred_lb = y_pred["Lower bound"]
    #     y_pred_ub = y_pred["Upper bound"]
    #     y_true_lb, y_true_ub = y_true[:, 1], y_true[:, 1]
    #     ev = {
    #         "Lower bound": {
    #             "Mean squared error": mean_squared_error(y_true_lb, y_pred_lb),
    #             "Explained variance": explained_variance_score(y_true_lb, y_pred_lb),
    #             "Max error": max_error(y_true_lb, y_pred_lb),
    #             "Mean absolute error": mean_absolute_error(y_true_lb, y_pred_lb),
    #             "R2": r2_score(y_true_lb, y_pred_lb),
    #             "Median absolute error": mean_absolute_error(y_true_lb, y_pred_lb),
    #         },
    #         "Upper bound": {
    #             "Mean squared error": mean_squared_error(y_true_ub, y_pred_ub),
    #             "Explained variance": explained_variance_score(y_true_ub, y_pred_ub),
    #             "Max error": max_error(y_true_ub, y_pred_ub),
    #             "Mean absolute error": mean_absolute_error(y_true_ub, y_pred_ub),
    #             "R2": r2_score(y_true_ub, y_pred_ub),
    #             "Median absolute error": mean_absolute_error(y_true_ub, y_pred_ub),
    #         },
    #     }
    #     return ev

    def sample_predict(
        self,
        features: Features,
        sample: TrainingSample,
    ) -> Dict[str, float]:
        pred: Dict[str, float] = {}
        x, _ = self.sample_xy(features, sample)
        if self.lb_regressor is not None:
            lb_pred = self.lb_regressor.predict(np.array(x["Lower bound"]))
            pred["Lower bound"] = lb_pred[0, 0]
        else:
            logger.info("Lower bound regressor not fitted. Skipping.")
        if self.ub_regressor is not None:
            ub_pred = self.ub_regressor.predict(np.array(x["Upper bound"]))
            pred["Upper bound"] = ub_pred[0, 0]
        else:
            logger.info("Upper bound regressor not fitted. Skipping.")
        return pred

    @staticmethod
    def sample_xy(
        features: Features,
        sample: TrainingSample,
    ) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]]]:
        x: Dict[str, List[List[float]]] = {}
        y: Dict[str, List[List[float]]] = {}
        f = list(features["Instance"]["User features"])
        if "LP value" in sample and sample["LP value"] is not None:
            f += [sample["LP value"]]
        x["Lower bound"] = [f]
        x["Upper bound"] = [f]
        if "Lower bound" in sample and sample["Lower bound"] is not None:
            y["Lower bound"] = [[sample["Lower bound"]]]
        if "Upper bound" in sample and sample["Upper bound"] is not None:
            y["Upper bound"] = [[sample["Upper bound"]]]
        return x, y

    def sample_evaluate(
        self,
        features: Features,
        sample: TrainingSample,
    ) -> Dict[Hashable, Dict[str, float]]:
        def compare(y_pred: float, y_actual: float) -> Dict[str, float]:
            err = np.round(abs(y_pred - y_actual), 8)
            return {
                "Actual value": y_actual,
                "Predicted value": y_pred,
                "Absolute error": err,
                "Relative error": err / y_actual,
            }

        result: Dict[Hashable, Dict[str, float]] = {}
        pred = self.sample_predict(features, sample)
        if "Upper bound" in sample and sample["Upper bound"] is not None:
            result["Upper bound"] = compare(pred["Upper bound"], sample["Upper bound"])
        if "Lower bound" in sample and sample["Lower bound"] is not None:
            result["Lower bound"] = compare(pred["Lower bound"], sample["Lower bound"])
        return result
