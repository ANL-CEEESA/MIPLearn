# MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class WarmStartPredictor:
    def __init__(self,
                 thr_fix_zero=0.05,
                 thr_fix_one=0.95,
                 thr_predict=0.95):
        self.model = None
        self.thr_predict = thr_predict
        self.thr_fix_zero = thr_fix_zero
        self.thr_fix_one = thr_fix_one

    def fit(self, x_train, y_train):
        assert isinstance(x_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert y_train.shape[1] == 2
        assert y_train.shape[0] == x_train.shape[0]
        y_hat = np.average(y_train[:, 1])
        if y_hat < self.thr_fix_zero or y_hat > self.thr_fix_one:
            self.model = int(y_hat)
        else:
            self.model = make_pipeline(StandardScaler(), LogisticRegression())
            self.model.fit(x_train, y_train[:, 1].astype(int))

    def predict(self, x_test):
        assert isinstance(x_test, np.ndarray)
        if isinstance(self.model, int):
            p_test = np.array([[1 - self.model, self.model]
                               for _ in range(x_test.shape[0])])
        else:
            p_test = self.model.predict_proba(x_test)
        p_test[p_test < self.thr_predict] = 0
        p_test[p_test > 0] = 1
        p_test = p_test.astype(int)
        return p_test
