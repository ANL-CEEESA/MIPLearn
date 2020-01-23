# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import miplearn
import numpy as np
import pyomo.environ as pe


class KnapsackInstance(miplearn.Instance):
    def __init__(self, weights, prices, capacity):
        self.weights = weights
        self.prices = prices
        self.capacity = capacity

    def to_model(self):
        model = pe.ConcreteModel()
        items = range(len(self.weights))
        model.x = pe.Var(items, domain=pe.Binary)
        model.OBJ = pe.Objective(rule=lambda m: sum(m.x[v] * self.prices[v] for v in items),
                                 sense=pe.maximize)
        model.eq_capacity = pe.Constraint(rule=lambda m: sum(m.x[v] * self.weights[v]
                                                             for v in items) <= self.capacity)
        return model

    def get_instance_features(self):
        return np.array([
            self.capacity,
            np.average(self.weights),
        ])

    def get_variable_features(self, var, index):
        return np.array([
            self.weights[index],
            self.prices[index],
        ])


class KnapsackInstance2(KnapsackInstance):
    """
    Alternative implementation of the Knapsack Problem, which assigns a different category for each
    decision variable, and therefore trains one machine learning model per variable.
    """

    def get_instance_features(self):
        return np.hstack([self.weights, self.prices])

    def get_variable_features(self, var, index):
        return np.array([
        ])

    def get_variable_category(self, var, index):
        return index
