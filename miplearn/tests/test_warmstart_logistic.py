# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn.warmstart import LogisticWarmStartPredictor
from sklearn.metrics import accuracy_score, precision_score
import numpy as np


def _generate_dataset(ground_truth, n_samples=10_000):
    x_train = np.random.rand(n_samples,5)
    x_test = np.random.rand(n_samples,5)
    y_train = ground_truth(x_train)
    y_test = ground_truth(x_test)
    return x_train, y_train, x_test, y_test


def _is_sum_greater_than_two(x):
    y = (np.sum(x, axis=1) > 2.0).astype(int)
    return np.vstack([y, 1 - y]).transpose()


def _always_zero(x):
    y = np.zeros((1, x.shape[0]))
    return np.vstack([y, 1 - y]).transpose()


def _random_values(x):
    y = np.random.randint(2, size=x.shape[0])
    return np.vstack([y, 1 - y]).transpose()
    
    
def test_logistic_ws_with_balanced_labels():
    x_train, y_train, x_test, y_test = _generate_dataset(_is_sum_greater_than_two)
    ws = LogisticWarmStartPredictor()
    ws.fit(x_train, y_train)
    y_pred = ws.predict(x_test)
    assert accuracy_score(y_test[:,0], y_pred[:,0]) > 0.99
    assert accuracy_score(y_test[:,1], y_pred[:,1]) > 0.99
    
    
def test_logistic_ws_with_unbalanced_labels():
    x_train, y_train, x_test, y_test = _generate_dataset(_always_zero)
    ws = LogisticWarmStartPredictor()
    ws.fit(x_train, y_train)
    y_pred = ws.predict(x_test)
    assert accuracy_score(y_test[:,0], y_pred[:,0]) == 1.0
    assert accuracy_score(y_test[:,1], y_pred[:,1]) == 1.0

    
def test_logistic_ws_with_unpredictable_labels():
    x_train, y_train, x_test, y_test = _generate_dataset(_random_values)
    ws = LogisticWarmStartPredictor()
    ws.fit(x_train, y_train)
    y_pred = ws.predict(x_test)
    assert np.sum(y_pred) == 0

    
def test_logistic_ws_with_small_sample_size():
    x_train, y_train, x_test, y_test = _generate_dataset(_random_values, n_samples=3)
    ws = LogisticWarmStartPredictor()
    ws.fit(x_train, y_train)
    y_pred = ws.predict(x_test)
    assert np.sum(y_pred) == 0
