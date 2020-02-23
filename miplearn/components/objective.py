#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

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
    def __init__(self,
                 regressor=LinearRegression()):
        self.ub_regressor = None
        self.lb_regressor = None
        self.regressor_prototype = regressor
    
    def before_solve(self, solver, instance, model):
        if self.ub_regressor is not None:
            lb, ub = self.predict([instance])[0]
            instance.predicted_ub = ub
            instance.predicted_lb = lb
            logger.info("Predicted objective: [%.2f, %.2f]" % (lb, ub))
    
    def after_solve(self, solver, instance, model):
        pass
    
    def merge(self, other):
        pass
    
    def fit(self, training_instances):
        features = InstanceFeaturesExtractor().extract(training_instances)
        ub = ObjectiveValueExtractor(kind="upper bound").extract(training_instances)
        lb = ObjectiveValueExtractor(kind="lower bound").extract(training_instances)
        self.ub_regressor = deepcopy(self.regressor_prototype)
        self.lb_regressor = deepcopy(self.regressor_prototype)
        self.ub_regressor.fit(features, ub)
        self.lb_regressor.fit(features, lb)
        
    def predict(self, instances):
        features = InstanceFeaturesExtractor().extract(instances)
        lb = self.lb_regressor.predict(features)
        ub = self.ub_regressor.predict(features)
        return np.hstack([lb, ub])
