# MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

# from .warmstart import WarmStartPredictor
from .transformers import PerVariableTransformer
from .warmstart import WarmStartPredictor
import pyomo.environ as pe
import numpy as np


class LearningSolver:
    """
    Mixed-Integer Linear Programming (MIP) solver that extracts information from previous runs,
    using Machine Learning methods, to accelerate the solution of new (yet unseen) instances.
    """

    def __init__(self,
                 threads=4,
                 parent_solver=pe.SolverFactory('cbc')):
        self.parent_solver = parent_solver
        self.parent_solver.options["threads"] = threads
        self.x_train = {}
        self.y_train = {}
        self.ws_predictors = {}

    def solve(self, instance, tee=False):
        # Convert instance into concrete model
        model = instance.to_model()

        # Split decision variables according to their category
        transformer = PerVariableTransformer()
        var_split = transformer.split_variables(instance, model)

        # Build x_test and update x_train
        x_test = {}
        for category in var_split.keys():
            var_index_pairs = var_split[category]
            x = transformer.transform_instance(instance, var_index_pairs)
            x_test[category] = x
            if category not in self.x_train.keys():
                self.x_train[category] = x
            else:
                self.x_train[category] = np.vstack([self.x_train[category], x])

        # Predict warm start
        for category in var_split.keys():
            if category in self.ws_predictors.keys():
                var_index_pairs = var_split[category]
                ws = self.ws_predictors[category].predict(x_test[category])
                assert ws.shape == (len(var_index_pairs), 2)
                for i in range(len(var_index_pairs)):
                    var, index = var_index_pairs[i]
                    if ws[i,0] == 1:
                        var[index].value = 1
                    elif ws[i,1] == 1:
                        var[index].value = 0

        # Solve MILP
        self.parent_solver.solve(model, tee=tee, warmstart=True)

        # Update y_train
        for category in var_split.keys():
            var_index_pairs = var_split[category]
            y = transformer.transform_solution(var_index_pairs)
            if category not in self.y_train.keys():
                self.y_train[category] = y
            else:
                self.y_train[category] = np.vstack([self.y_train[category], y])

    def fit(self, x_train_dict=None, y_train_dict=None):
        if x_train_dict is None:
            x_train_dict = self.x_train
            y_train_dict = self.y_train
        for category in x_train_dict.keys():
            x_train = x_train_dict[category]
            y_train = y_train_dict[category]
            self.ws_predictors[category] = WarmStartPredictor()
            self.ws_predictors[category].fit(x_train, y_train)

    def _solve(self, tee):
        self.parent_solver.solve(tee=tee)