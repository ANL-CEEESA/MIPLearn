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
                 thr_balance=1.0,
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

        
class WarmStartComponent(Component):
    def __init__(self,
                 predictor=AdaptivePredictor(),
                 mode="exact",
                 max_fpr=[0.01, 0.01],
                 min_threshold=[0.75, 0.75],
                 dynamic_thresholds=False,
                ):
        self.mode = mode
        self.x_train = {}
        self.y_train = {}
        self.predictors = {}
        self.is_warm_start_available = False
        self.max_fpr = max_fpr
        self.min_threshold = min_threshold
        self.thresholds = {}
        self.predictor_factory = predictor
        self.dynamic_thresholds = dynamic_thresholds
        
    
    def before_solve(self, solver, instance, model):
#         # Solve linear relaxation
#         lr_solver = pe.SolverFactory("gurobi")
#         lr_solver.options["threads"] = 4
#         lr_solver.options["relax_integrality"] = 1
#         lr_solver.solve(model, tee=solver.tee)
        
        # Build x_test
        x_test = CombinedExtractor([UserFeaturesExtractor(),
                                    SolutionExtractor(),
                                   ]).extract([instance], [model])
        
        # Update self.x_train
        self.x_train = Extractor.merge([self.x_train, x_test],
                                       vertical=True)
        
        # Predict solutions
        count_total, count_fixed = 0, 0
        var_split = Extractor.split_variables(instance, model)
        for category in var_split.keys():
            var_index_pairs = var_split[category]
            
            # Clear current values
            for i in range(len(var_index_pairs)):
                var, index = var_index_pairs[i]
                var[index].value = None
            
            # Make predictions
            for label in [0,1]:
                if (category, label) not in self.predictors.keys():
                    continue
                ws = self.predictors[category, label].predict_proba(x_test[category])
                assert ws.shape == (len(var_index_pairs), 2)
                for i in range(len(var_index_pairs)):
                    count_total += 1
                    var, index = var_index_pairs[i]
                    logger.debug("%s[%s] ws=%.6f threshold=%.6f" % (var, index, ws[i, 1], self.thresholds[category, label]))
                    if ws[i, 1] > self.thresholds[category, label]:
                        logger.debug("Setting %s[%s] to %d" % (var, index, label))
                        count_fixed += 1
                        if self.mode == "heuristic":
                            var[index].fix(label)
                            if solver.is_persistent:
                                solver.internal_solver.update_var(var[index])
                        else:
                            var[index].value = label
                            self.is_warm_start_available = True
                            
            # Clear current values
            for i in range(len(var_index_pairs)):
                var, index = var_index_pairs[i]
                if var[index].value is None:
                    logger.debug("Variable %s[%s] not set" % (var, index))
                else:
                    logger.debug("Varible %s[%s] set to %.2f" % (var, index, var[index].value))

                
        logger.info("Setting values for %d variables (out of %d)" % (count_fixed, count_total // 2))


    def after_solve(self, solver, instance, model):
        y_test = SolutionExtractor().extract([instance], [model])
        self.y_train = Extractor.merge([self.y_train, y_test], vertical=True)        
                
    def fit(self, solver, n_jobs=1):
        for category in tqdm(self.x_train.keys(), desc="Fit (warm start)"):
            x_train = self.x_train[category]
            y_train = self.y_train[category]
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
                logger.debug("    Setting threshold to %.4f (fpr=%.4f, tpr=%.4f)" % (thresholds[k], fpr[k], tpr[k]))
                self.thresholds[category, label] = thresholds[k]
                

    def merge(self, other_components):
        # Merge x_train and y_train
        keys = set(self.x_train.keys())
        for comp in other_components:
            keys = keys.union(set(comp.x_train.keys()))
        for key in keys:
            x_train_submatrices = [comp.x_train[key]
                                   for comp in other_components
                                   if key in comp.x_train.keys()]
            y_train_submatrices = [comp.y_train[key]
                                   for comp in other_components
                                   if key in comp.y_train.keys()]
            if key in self.x_train.keys():
                x_train_submatrices += [self.x_train[key]]
                y_train_submatrices += [self.y_train[key]]
            self.x_train[key] = np.vstack(x_train_submatrices)
            self.y_train[key] = np.vstack(y_train_submatrices)

        # Merge trained predictors
        for comp in other_components:
            for key in comp.predictors.keys():
                if key not in self.predictors.keys():
                    self.predictors[key] = comp.predictors[key]
                    self.thresholds[key] = comp.thresholds[key]

                    
# Deprecated               
class WarmStartPredictor(ABC):
    def __init__(self, thr_clip=[0.50, 0.50]):
        self.models = [None, None]
        self.thr_clip = thr_clip
        
    def fit(self, x_train, y_train):
        assert isinstance(x_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        y_train = y_train.astype(int)
        assert y_train.shape[0] == x_train.shape[0]
        assert y_train.shape[1] == 2
        for i in [0,1]:
            self.models[i] = self._fit(x_train, y_train[:, i],  i)

    def predict(self, x_test):
        assert isinstance(x_test, np.ndarray)
        y_pred = np.zeros((x_test.shape[0], 2))
        for i in [0,1]:
            if isinstance(self.models[i], int):
                y_pred[:, i] = self.models[i]
            else:
                y = self.models[i].predict_proba(x_test)[:,1]
                y[y < self.thr_clip[i]] = 0.
                y[y > 0.] = 1.
                y_pred[:, i] = y
        return y_pred.astype(int)

    @abstractmethod
    def _fit(self, x_train, y_train, label):
        pass

    
# Deprecated               
class LogisticWarmStartPredictor(WarmStartPredictor):
    def __init__(self,
                 min_samples=100,
                 thr_fix=[0.99, 0.99],
                 thr_balance=[0.80, 0.80],
                 thr_alpha=[0.50, 0.50],
                ):
        super().__init__()
        self.min_samples = min_samples
        self.thr_fix = thr_fix
        self.thr_balance = thr_balance
        self.thr_alpha = thr_alpha

    def _fit(self, x_train, y_train, label):
        y_train_avg = np.average(y_train)

        # If number of samples is too small, don't predict anything.
        if x_train.shape[0] < self.min_samples:
            return 0
        
        # If vast majority of observations are true, always return true.
        if y_train_avg > self.thr_fix[label]:
            return 1
        
        # If dataset is not balanced enough, don't predict anything.
        if y_train_avg < (1 - self.thr_balance[label]) or y_train_avg > self.thr_balance[label]:
            return 0
            
        reg = make_pipeline(StandardScaler(), LogisticRegression())
        reg_score = np.mean(cross_val_score(reg, x_train, y_train, cv=5))
        dummy_score = max(y_train_avg, 1 - y_train_avg)
        reg_thr = 1. * self.thr_alpha[label] + dummy_score * (1 - self.thr_alpha[label])

        # If cross-validation score is too low, don't predict anything.
        if reg_score < reg_thr:
            return 0
        
        reg.fit(x_train, y_train.astype(int))
        return reg
    
    
# Deprecated
class KnnWarmStartPredictor(WarmStartPredictor):
    def __init__(self,
                 k=50,
                 min_samples=1,
                 thr_clip=[0.80, 0.80],
                 thr_fix=[1.0, 1.0],
                ):
        super().__init__(thr_clip=thr_clip)
        self.k = k
        self.thr_fix = thr_fix
        self.min_samples = min_samples

    def _fit(self, x_train, y_train, label):
        y_train_avg = np.average(y_train)

        # If number of training samples is too small, don't predict anything.
        if x_train.shape[0] < self.min_samples:
            logger.debug("Too few samples; return 0")
            return 0
        
        # If vast majority of observations are true, always return true.
        if y_train_avg >= self.thr_fix[label]:
            logger.debug("Consensus reached; return 1")
            return 1
        
        # If vast majority of observations are false, always return false.
        if y_train_avg <= (1 - self.thr_fix[label]):
            logger.debug("Consensus reached; return 0")
            return 0
        
        logger.debug("Training classifier...")
        k = min(self.k, x_train.shape[0])
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        return knn
    
    
    
                