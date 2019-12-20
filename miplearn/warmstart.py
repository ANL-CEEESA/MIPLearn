# MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
import numpy as np

class WarmStartPredictor:
    def __init__(self, model=None, threshold=0.80):
        self.model = model
        self.threshold = threshold
    
    def fit(self, train_x, train_y):
        pass
    
    def predict(self, x):
        if self.model is None: return None
        assert isinstance(x, np.ndarray)
        y = self.model.predict(x)
        n_vars = y.shape[0]
        ws = np.array([float("nan")] * n_vars)
        ws[y[:,0] > self.threshold] = 1.0
        ws[y[:,1] > self.threshold] = 0.0
        return ws
        
    