# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import KnnWarmStartPredictor
from sklearn.metrics import accuracy_score, precision_score
import numpy as np


def test_knn_with_consensus():
    x_train = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [1.0, 1.0],
    ])
    y_train = np.array([
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [1., 0.],
    ])
    ws = KnnWarmStartPredictor(k=3, thr_clip=[0.75, 0.75])
    ws.fit(x_train, y_train)
    
    x_test = np.array([[0.0, 0.0]])
    y_test = np.array([[0, 1]])
    assert (ws.predict(x_test) == y_test).all()
    
def test_knn_without_consensus():
    x_train = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.9, 0.9],
        [1.0, 1.0],
    ])
    y_train = np.array([
        [0., 1.],
        [0., 1.],
        [1., 0.],
        [1., 0.],
    ])
    ws = KnnWarmStartPredictor(k=4, thr_clip=[0.75, 0.75])
    ws.fit(x_train, y_train)
    
    x_test = np.array([[0.5, 0.5]])
    y_test = np.array([[0, 0]])
    assert (ws.predict(x_test) == y_test).all()    
    
def test_knn_always_true():
    x_train = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.9, 0.9],
        [1.0, 1.0],
    ])
    y_train = np.array([
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
    ])
    ws = KnnWarmStartPredictor(k=4, thr_clip=[0.75, 0.75])
    ws.fit(x_train, y_train)
    
    x_test = np.array([[0.5, 0.5]])
    y_test = np.array([[1, 0]])
    assert (ws.predict(x_test) == y_test).all()       
    
def test_knn_always_false():
    x_train = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.9, 0.9],
        [1.0, 1.0],
    ])
    y_train = np.array([
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
    ])
    ws = KnnWarmStartPredictor(k=4, thr_clip=[0.75, 0.75])
    ws.fit(x_train, y_train)
    
    x_test = np.array([[0.5, 0.5]])
    y_test = np.array([[0, 1]])
    assert (ws.predict(x_test) == y_test).all()           