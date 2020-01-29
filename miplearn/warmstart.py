# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from . import Component
from .transformers import PerVariableTransformer

from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


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
                 thr_fix=[0.95, 0.95],
                 thr_balance=[0.95, 0.95],
                 thr_score=[0.95, 0.95]):
        super().__init__()
        self.min_samples = min_samples
        self.thr_fix = thr_fix
        self.thr_balance = thr_balance
        self.thr_score = thr_score

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

        # If cross-validation score is too low, don't predict anything.
        if reg_score < self.thr_score[label]:
            return 0
        
        reg.fit(x_train, y_train.astype(int))
        return reg
    
    
class KnnWarmStartPredictor(WarmStartPredictor):
    def __init__(self, k=50,
                 thr_clip=[0.90, 0.90],
                 thr_fix=[0.99, 0.99]):
        super().__init__(thr_clip=thr_clip)
        self.k = k
        self.thr_fix = thr_fix

    def _fit(self, x_train, y_train, label):
        y_train_avg = np.average(y_train)

        # If number of training samples is too small, don't predict anything.
        if x_train.shape[0] < self.k:
            return 0
        
        # If vast majority of observations are true, always return true.
        if y_train_avg > self.thr_fix[label]:
            return 1
        
        # If vast majority of observations are false, always return false.
        if y_train_avg < (1 - self.thr_fix[label]):
            return 0
        
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(x_train, y_train)
        return knn
    
    
class WarmStartComponent(Component):
    def __init__(self,
                 predictor_prototype=LogisticWarmStartPredictor(),
                 mode="exact",
                ):
        self.mode = mode
        self.transformer = PerVariableTransformer()
        self.x_train = {}
        self.y_train = {}
        self.predictors = {}
        self.predictor_prototype = predictor_prototype
    
    def before_solve(self, solver, instance, model):
        var_split = self.transformer.split_variables(instance, model)
        x_test = {}
        
        # Collect training data (x_train) and build x_test
        for category in var_split.keys():
            var_index_pairs = var_split[category]
            x = self.transformer.transform_instance(instance, var_index_pairs)
            x_test[category] = x
            if category not in self.x_train.keys():
                self.x_train[category] = x
            else:
                assert x.shape[1] == self.x_train[category].shape[1]
                self.x_train[category] = np.vstack([self.x_train[category], x])
        
        # Predict solutions
        for category in var_split.keys():
            var_index_pairs = var_split[category]
            if category in self.predictors.keys():
                ws = self.predictors[category].predict(x_test[category])
                assert ws.shape == (len(var_index_pairs), 2)
                for i in range(len(var_index_pairs)):
                    var, index = var_index_pairs[i]
                    if self.mode == "heuristic":
                        if ws[i,0] == 1:
                            var[index].fix(0)
                            if solver.is_persistent:
                                solver.internal_solver.update_var(var[index])
                        elif ws[i,1] == 1:
                            var[index].fix(1)
                            if solver.is_persistent:
                                solver.internal_solver.update_var(var[index])
                    else:
                        if ws[i,0] == 1:
                            var[index].value = 0
                        elif ws[i,1] == 1:
                            var[index].value = 1

    def after_solve(self, solver, instance, model):
        var_split = self.transformer.split_variables(instance, model)
        for category in var_split.keys():
            var_index_pairs = var_split[category]
            y = self.transformer.transform_solution(var_index_pairs)
            if category not in self.y_train.keys():
                self.y_train[category] = y
            else:
                self.y_train[category] = np.vstack([self.y_train[category], y])
                
    def fit(self, solver):
        for category in self.x_train.keys():
            x_train = self.x_train[category]
            y_train = self.y_train[category]
            self.predictors[category] = deepcopy(self.predictor_prototype)
            self.predictors[category].fit(x_train, y_train)

    def merge(self, other):
        for c in other.x_train.keys():
            if c not in self.x_train:
                self.x_train[c] = other.x_train[c]
                self.y_train[c] = other.y_train[c]
            else:
                self.x_train[c] = np.vstack([self.x_train[c], other.x_train[c]])
                self.y_train[c] = np.vstack([self.y_train[c], other.y_train[c]])
            if (c in other.predictors.keys()) and (c not in self.predictors.keys()):
                self.predictors[c] = other.predictors[c]