# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from abc import ABC, abstractmethod
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

class WarmStartPredictor(ABC):
    def __init__(self):
        self.models = [None, None]
        
    def fit(self, x_train, y_train):
        assert isinstance(x_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert y_train.shape[0] == x_train.shape[0]
        assert y_train.shape[1] == 2
        for i in [0,1]:
            self.models[i] = self._fit(x_train, y_train[:, i],  i)

    def predict(self, x_test):
        assert isinstance(x_test, np.ndarray)
        y_pred = np.zeros((x_test.shape[0], 2), dtype=np.int)
        for i in [0,1]:
            if isinstance(self.models[i], int):
                y_pred[:, i] = self.models[i]
            else:
                y_pred[:, i] = self.models[i].predict(x_test)
        return y_pred

    @abstractmethod
    def _fit(self, x_train, y_train, label):
        pass


class LogisticWarmStartPredictor(WarmStartPredictor):
    def __init__(self,
                 min_samples=100,
                 thr_fix=[0.99, 0.99],
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