#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from sklearn.metrics import (
    mean_squared_error,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    r2_score,
)

from .. import Component, InstanceFeaturesExtractor, ObjectiveValueExtractor
from sklearn.linear_model import LinearRegression
from copy import deepcopy
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ObjectiveValueComponent(Component):
    """
    A Component which predicts the optimal objective value of the problem.
    """

    def __init__(self, regressor=LinearRegression()):
        self.ub_regressor = None
        self.lb_regressor = None
        self.regressor_prototype = regressor

    def before_solve(self, solver, instance, model):
        if self.ub_regressor is not None:
            logger.info("Predicting optimal value...")
            lb, ub = self.predict([instance])[0]
            instance.predicted_ub = ub
            instance.predicted_lb = lb
            logger.info("Predicted values: lb=%.2f, ub=%.2f" % (lb, ub))

    def after_solve(self, solver, instance, model, results):
        if self.ub_regressor is not None:
            results["Predicted UB"] = instance.predicted_ub
            results["Predicted LB"] = instance.predicted_lb
        else:
            results["Predicted UB"] = None
            results["Predicted LB"] = None

    def fit(self, training_instances):
        logger.debug("Extracting features...")
        features = InstanceFeaturesExtractor().extract(training_instances)
        ub = ObjectiveValueExtractor(kind="upper bound").extract(training_instances)
        lb = ObjectiveValueExtractor(kind="lower bound").extract(training_instances)
        assert ub.shape == (len(training_instances), 1)
        assert lb.shape == (len(training_instances), 1)
        self.ub_regressor = deepcopy(self.regressor_prototype)
        self.lb_regressor = deepcopy(self.regressor_prototype)
        logger.debug("Fitting ub_regressor...")
        self.ub_regressor.fit(features, ub.ravel())
        logger.debug("Fitting ub_regressor...")
        self.lb_regressor.fit(features, lb.ravel())

    def predict(self, instances):
        features = InstanceFeaturesExtractor().extract(instances)
        lb = self.lb_regressor.predict(features)
        ub = self.ub_regressor.predict(features)
        assert lb.shape == (len(instances),)
        assert ub.shape == (len(instances),)
        return np.array([lb, ub]).T

    def evaluate(self, instances):
        y_pred = self.predict(instances)
        y_true = np.array([[inst.lower_bound, inst.upper_bound] for inst in instances])
        y_true_lb, y_true_ub = y_true[:, 0], y_true[:, 1]
        y_pred_lb, y_pred_ub = y_pred[:, 1], y_pred[:, 1]
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
