#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import List, Dict, Any, TYPE_CHECKING, Tuple, Hashable

import numpy as np
from sklearn.linear_model import LinearRegression

from miplearn.classifiers import Regressor
from miplearn.classifiers.sklearn import ScikitLearnRegressor
from miplearn.components.component import Component
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
        self.regressors: Dict[str, Regressor] = {}
        self.regressor_prototype = regressor

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
        for (c, v) in pred.items():
            logger.info(f"Predicted {c.lower()}: %.6e" % v)
            stats[f"Objective: Predicted {c.lower()}"] = v  # type: ignore

    def fit_xy(
        self,
        x: Dict[Hashable, np.ndarray],
        y: Dict[Hashable, np.ndarray],
    ) -> None:
        for c in ["Upper bound", "Lower bound"]:
            if c in y:
                self.regressors[c] = self.regressor_prototype.clone()
                self.regressors[c].fit(x[c], y[c])

    def sample_predict(
        self,
        features: Features,
        sample: TrainingSample,
    ) -> Dict[str, float]:
        pred: Dict[str, float] = {}
        x, _ = self.sample_xy(features, sample)
        for c in ["Upper bound", "Lower bound"]:
            if c in self.regressors is not None:
                pred[c] = self.regressors[c].predict(np.array(x[c]))[0, 0]
            else:
                logger.info(f"{c} regressor not fitted. Skipping.")
        return pred

    @staticmethod
    def sample_xy(
        features: Features,
        sample: TrainingSample,
    ) -> Tuple[Dict[Hashable, List[List[float]]], Dict[Hashable, List[List[float]]]]:
        assert features.instance is not None
        assert features.instance.user_features is not None
        x: Dict[Hashable, List[List[float]]] = {}
        y: Dict[Hashable, List[List[float]]] = {}
        f = list(features.instance.user_features)
        if sample.lp_value is not None:
            f += [sample.lp_value]
        x["Upper bound"] = [f]
        x["Lower bound"] = [f]
        if sample.lower_bound is not None:
            y["Lower bound"] = [[sample.lower_bound]]
        if sample.upper_bound is not None:
            y["Upper bound"] = [[sample.upper_bound]]
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
        if sample.upper_bound is not None:
            result["Upper bound"] = compare(pred["Upper bound"], sample.upper_bound)
        if sample.lower_bound is not None:
            result["Lower bound"] = compare(pred["Lower bound"], sample.lower_bound)
        return result
