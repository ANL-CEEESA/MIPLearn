#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import List, Dict, Any, TYPE_CHECKING, Tuple, Hashable, Optional

import numpy as np
from overrides import overrides
from sklearn.linear_model import LinearRegression

from miplearn.classifiers import Regressor
from miplearn.classifiers.sklearn import ScikitLearnRegressor
from miplearn.components.component import Component
from miplearn.features import Sample
from miplearn.instance.base import Instance
from miplearn.types import LearningSolveStats

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

    @overrides
    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        logger.info("Predicting optimal value...")
        pred = self.sample_predict(sample)
        for (c, v) in pred.items():
            logger.info(f"Predicted {c.lower()}: %.6e" % v)
            stats[f"Objective: Predicted {c.lower()}"] = v  # type: ignore

    @overrides
    def fit_xy(
        self,
        x: Dict[Hashable, np.ndarray],
        y: Dict[Hashable, np.ndarray],
    ) -> None:
        for c in ["Upper bound", "Lower bound"]:
            if c in y:
                self.regressors[c] = self.regressor_prototype.clone()
                self.regressors[c].fit(x[c], y[c])

    def sample_predict(self, sample: Sample) -> Dict[str, float]:
        pred: Dict[str, float] = {}
        x, _ = self.sample_xy(None, sample)
        for c in ["Upper bound", "Lower bound"]:
            if c in self.regressors is not None:
                pred[c] = self.regressors[c].predict(np.array(x[c]))[0, 0]
            else:
                logger.info(f"{c} regressor not fitted. Skipping.")
        return pred

    @overrides
    def sample_xy(
        self,
        _: Optional[Instance],
        sample: Sample,
    ) -> Tuple[Dict[Hashable, List[List[float]]], Dict[Hashable, List[List[float]]]]:
        # Instance features
        assert sample.after_load is not None
        assert sample.after_load.instance is not None
        f = sample.after_load.instance.to_list()

        # LP solve features
        if sample.after_lp is not None:
            assert sample.after_lp.lp_solve is not None
            f.extend(sample.after_lp.lp_solve.to_list())

        # Features
        x: Dict[Hashable, List[List[float]]] = {
            "Upper bound": [f],
            "Lower bound": [f],
        }

        # Labels
        y: Dict[Hashable, List[List[float]]] = {}
        if sample.after_mip is not None:
            mip_stats = sample.after_mip.mip_solve
            assert mip_stats is not None
            if mip_stats.mip_lower_bound is not None:
                y["Lower bound"] = [[mip_stats.mip_lower_bound]]
            if mip_stats.mip_upper_bound is not None:
                y["Upper bound"] = [[mip_stats.mip_upper_bound]]

        return x, y

    @overrides
    def sample_evaluate(
        self,
        instance: Instance,
        sample: Sample,
    ) -> Dict[Hashable, Dict[str, float]]:
        assert sample.after_mip is not None
        assert sample.after_mip.mip_solve is not None

        def compare(y_pred: float, y_actual: float) -> Dict[str, float]:
            err = np.round(abs(y_pred - y_actual), 8)
            return {
                "Actual value": y_actual,
                "Predicted value": y_pred,
                "Absolute error": err,
                "Relative error": err / y_actual,
            }

        result: Dict[Hashable, Dict[str, float]] = {}
        pred = self.sample_predict(sample)
        actual_ub = sample.after_mip.mip_solve.mip_upper_bound
        actual_lb = sample.after_mip.mip_solve.mip_lower_bound
        if actual_ub is not None:
            result["Upper bound"] = compare(pred["Upper bound"], actual_ub)
        if actual_lb is not None:
            result["Lower bound"] = compare(pred["Lower bound"], actual_lb)
        return result
