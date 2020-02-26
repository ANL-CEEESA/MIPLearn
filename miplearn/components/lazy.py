#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from .component import Component
from ..extractors import *

from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from tqdm.auto import tqdm
import pyomo.environ as pe
import logging
logger = logging.getLogger(__name__)


class LazyConstraintsComponent(Component):
    """
    A component that predicts which lazy constraints to enforce.
    """
    
    def __init__(self):
        self.violations = set()
        self.count = {}
        self.n_samples = 0
    
    def before_solve(self, solver, instance, model):
        logger.info("Enforcing %d lazy constraints" % len(self.violations))
        for v in self.violations:
            if self.count[v] < self.n_samples * 0.05:
                continue
            cut = instance.build_lazy_constraint(model, v)
            solver.internal_solver.add_constraint(cut)
        
    def after_solve(self, solver, instance, model, results):
        pass
                
    def fit(self, training_instances):
        logger.debug("Fitting...")
        self.n_samples = len(training_instances)
        for instance in training_instances:
            if not hasattr(instance, "found_violations"):
                continue
            for v in instance.found_violations:
                self.violations.add(v)
                if v not in self.count.keys():
                    self.count[v] = 0
                self.count[v] += 1
                
    def predict(self, instance, model=None):
        return self.violations
