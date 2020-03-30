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


class AdaptivePredictor:
    def __init__(self,
                 predictor=None,
                 min_samples_predict=1,
                 min_samples_cv=100,
                 thr_fix=0.999,
                 thr_alpha=0.50,
                 thr_balance=0.95,
                ):
        self.min_samples_predict = min_samples_predict
        self.min_samples_cv = min_samples_cv
        self.thr_fix = thr_fix
        self.thr_alpha = thr_alpha
        self.thr_balance = thr_balance
        self.predictor_factory = predictor
        
    def fit(self, x_train, y_train):
        n_samples = x_train.shape[0]
        
        # If number of samples is too small, don't predict anything.
        if n_samples < self.min_samples_predict:
            logger.debug("    Too few samples (%d); always predicting false" % n_samples)
            self.predictor = 0
            return
        
        # If vast majority of observations are false, always return false.
        y_train_avg = np.average(y_train)
        if y_train_avg <= 1.0 - self.thr_fix:
            logger.debug("    Most samples are negative (%.3f); always returning false" % y_train_avg)
            self.predictor = 0
            return
        
        # If vast majority of observations are true, always return true.
        if y_train_avg >= self.thr_fix:
            logger.debug("    Most samples are positive (%.3f); always returning true" % y_train_avg)
            self.predictor = 1
            return
        
        # If classes are too unbalanced, don't predict anything.
        if y_train_avg < (1 - self.thr_balance) or y_train_avg > self.thr_balance:
            logger.debug("    Classes are too unbalanced (%.3f); always returning false" % y_train_avg)
            self.predictor = 0
            return

        # Select ML model if none is provided
        if self.predictor_factory is None:
            if n_samples < 30:
                self.predictor_factory = KNeighborsClassifier(n_neighbors=n_samples)
            else:
                self.predictor_factory = make_pipeline(StandardScaler(), LogisticRegression())
        
        # Create predictor
        if callable(self.predictor_factory):
            pred = self.predictor_factory()
        else:
            pred = deepcopy(self.predictor_factory)
        
        # Skip cross-validation if number of samples is too small
        if n_samples < self.min_samples_cv:
            logger.debug("    Too few samples (%d); skipping cross validation" % n_samples)
            self.predictor = pred
            self.predictor.fit(x_train, y_train)
            return
        
        # Calculate cross-validation score
        cv_score = np.mean(cross_val_score(pred, x_train, y_train, cv=5))
        dummy_score = max(y_train_avg, 1 - y_train_avg)
        cv_thr = 1. * self.thr_alpha + dummy_score * (1 - self.thr_alpha)

        # If cross-validation score is too low, don't predict anything.
        if cv_score < cv_thr:
            logger.debug("    Score is too low (%.3f < %.3f); always returning false" % (cv_score, cv_thr))
            self.predictor = 0
        else:
            logger.debug("    Score is acceptable (%.3f > %.3f); training classifier" % (cv_score, cv_thr))
            self.predictor = pred
            self.predictor.fit(x_train, y_train)
        
    def predict_proba(self, x_test):
        if isinstance(self.predictor, int):
            y_pred = np.zeros((x_test.shape[0], 2))
            y_pred[:, self.predictor] = 1.0
            return y_pred
        else:
            return self.predictor.predict_proba(x_test)

        
class PrimalSolutionComponent(Component):
    """
    A component that predicts primal solutions.
    """
    
    def __init__(self,
                 predictor=AdaptivePredictor(),
                 mode="exact",
                 max_fpr=[1e-3, 1e-3],
                 min_threshold=[0.75, 0.75],
                 dynamic_thresholds=True,
                ):
        self.mode = mode
        self.predictors = {}
        self.is_warm_start_available = False
        self.max_fpr = max_fpr
        self.min_threshold = min_threshold
        self.thresholds = {}
        self.predictor_factory = predictor
        self.dynamic_thresholds = dynamic_thresholds
    
    def before_solve(self, solver, instance, model):
        solution = self.predict(instance)
        if solution is None:
            return
        if self.mode == "heuristic":
            solver.internal_solver.fix(solution)
        else:
            solver.internal_solver.set_warm_start(solution)
        
    def after_solve(self, solver, instance, model, results):
        pass
                
    def fit(self, training_instances):
        logger.debug("Extracting features...")
        features = VariableFeaturesExtractor().extract(training_instances)
        solutions = SolutionExtractor().extract(training_instances)
        
        for category in tqdm(features.keys(), desc="Fit (Primal)"):
            x_train = features[category]
            y_train = solutions[category]
            for label in [0, 1]:
                logger.debug("Fitting predictors[%s, %s]:" % (category, label))
                
                if callable(self.predictor_factory):
                    pred = self.predictor_factory(category, label)
                else:
                    pred = deepcopy(self.predictor_factory)
                self.predictors[category, label] = pred
                y = y_train[:, label].astype(int)
                pred.fit(x_train, y)

                # If y is either always one or always zero, set fixed threshold
                y_avg = np.average(y)
                if (not self.dynamic_thresholds) or y_avg <= 0.001 or y_avg >= 0.999:
                    self.thresholds[category, label] = self.min_threshold[label]
                    logger.debug("    Setting threshold to %.4f" % self.min_threshold[label])
                    continue
                
                # Calculate threshold dynamically using ROC curve
                y_scores = pred.predict_proba(x_train)[:, 1]
                fpr, tpr, thresholds = roc_curve(y, y_scores)
                k = 0
                while True:
                    if (k + 1) > len(fpr):
                        break
                    if fpr[k + 1] > self.max_fpr[label]:
                        break
                    if thresholds[k + 1] < self.min_threshold[label]:
                        break
                    k = k + 1
                logger.debug("    Setting threshold to %.4f (fpr=%.4f, tpr=%.4f)"%
                             (thresholds[k], fpr[k], tpr[k]))
                self.thresholds[category, label] = thresholds[k]
                
                
    def predict(self, instance):
        x_test = VariableFeaturesExtractor().extract([instance])
        solution = {}
        var_split = Extractor.split_variables(instance)
        all_none = True
        for category in var_split.keys():
            for (i, (var, index)) in enumerate(var_split[category]):
                if var not in solution.keys():
                    solution[var] = {}
                solution[var][index] = None
                for label in [0, 1]:
                    if (category, label) not in self.predictors.keys():
                        continue
                    ws = self.predictors[category, label].predict_proba(x_test[category])
                    logger.debug("%s[%s] ws=%.6f threshold=%.6f" %
                                 (var, index, ws[i, 1], self.thresholds[category, label]))
                    if ws[i, 1] >= self.thresholds[category, label]:
                        solution[var][index] = label
                        if all_none:
                            all_none = False
        if all_none:
            return None
        return solution
