# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from . import Component
from .transformers import PerVariableTransformer
from .extractors import *

from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm.auto import tqdm
import logging
logger = logging.getLogger(__name__)

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
    
    
class WarmStartComponent(Component):
    def __init__(self,
                 predictor_prototype=KnnWarmStartPredictor(),
                 mode="exact",
                ):
        self.mode = mode
        self.transformer = PerVariableTransformer()
        self.x_train = {}
        self.y_train = {}
        self.predictors = {}
        self.predictor_prototype = predictor_prototype
        self.is_warm_start_available = False
    
    def before_solve(self, solver, instance, model):
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
            if category not in self.predictors.keys():
                continue
            ws = self.predictors[category].predict(x_test[category])
            assert ws.shape == (len(var_index_pairs), 2)
            for i in range(len(var_index_pairs)):
                var, index = var_index_pairs[i]
                count_total += 1
                if self.mode == "heuristic":
                    if ws[i,0] > 0.5:
                        var[index].fix(0)
                        count_fixed += 1
                        if solver.is_persistent:
                            solver.internal_solver.update_var(var[index])
                    elif ws[i,1] > 0.5:
                        var[index].fix(1)
                        count_fixed += 1
                        if solver.is_persistent:
                            solver.internal_solver.update_var(var[index])
                else:
                    var[index].value = None
                    if ws[i,0] > 0.5:
                        count_fixed += 1
                        var[index].value = 0
                        self.is_warm_start_available = True
                    elif ws[i,1] > 0.5:
                        count_fixed += 1
                        var[index].value = 1
                        self.is_warm_start_available = True
        logger.info("Setting values for %d variables (out of %d)" % (count_fixed, count_total))


    def after_solve(self, solver, instance, model):
        y_test = SolutionExtractor().extract([instance], [model])
        self.y_train = Extractor.merge([self.y_train, y_test], vertical=True)        
                
    def fit(self, solver, n_jobs=1):
        for category in tqdm(self.x_train.keys(), desc="Warm start"):
            x_train = self.x_train[category]
            y_train = self.y_train[category]
            self.predictors[category] = deepcopy(self.predictor_prototype)
            self.predictors[category].fit(x_train, y_train)

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
                