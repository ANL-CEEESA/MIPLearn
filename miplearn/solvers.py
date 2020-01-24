# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from .transformers import PerVariableTransformer
from .warmstart import KnnWarmStartPredictor
import pyomo.environ as pe
import numpy as np
from copy import copy, deepcopy
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing



class LearningSolver:
    """
    Mixed-Integer Linear Programming (MIP) solver that extracts information from previous runs,
    using Machine Learning methods, to accelerate the solution of new (yet unseen) instances.
    """

    def __init__(self,
                 threads=4,
                 parent_solver=pe.SolverFactory('cbc'),
                 ws_predictor=KnnWarmStartPredictor(),
                 mode="exact"):
        self.parent_solver = parent_solver
        self.parent_solver.options["threads"] = threads
        self.mode = mode
        self.x_train = {}
        self.y_train = {}
        self.ws_predictors = {}
        self.ws_predictor_prototype = ws_predictor

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
                    if self.mode == "heuristic":
                        if ws[i,0] == 1:
                            var[index].fix(0)
                        elif ws[i,1] == 1:
                            var[index].fix(1)
                    else:
                        if ws[i,0] == 1:
                            var[index].value = 0
                        elif ws[i,1] == 1:
                            var[index].value = 1

        # Solve MILP
        solve_results = self._solve(model, tee=tee)

        # Update y_train
        for category in var_split.keys():
            var_index_pairs = var_split[category]
            y = transformer.transform_solution(var_index_pairs)
            if category not in self.y_train.keys():
                self.y_train[category] = y
            else:
                self.y_train[category] = np.vstack([self.y_train[category], y])
                
        return solve_results
                
    def parallel_solve(self, instances, n_jobs=4, label="Solve"):
        def _process(instance):
            solver = deepcopy(self)
            results = solver.solve(instance)
            return {
                "x_train": solver.x_train,
                "y_train": solver.y_train,
                "results": results,
            }

        def _merge(results):
            categories = results[0]["x_train"].keys()
            x_entries = [np.vstack([r["x_train"][c] for r in results]) for c in categories]
            y_entries = [np.vstack([r["y_train"][c] for r in results]) for c in categories]
            x_train = dict(zip(categories, x_entries))
            y_train = dict(zip(categories, y_entries))
            results = [r["results"] for r in results]
            return x_train, y_train, results

        results = Parallel(n_jobs=n_jobs)(
            delayed(_process)(instance)
            for instance in tqdm(instances, desc=label)
        )
        
        x_train, y_train, results = _merge(results)
        self.x_train = x_train
        self.y_train = y_train
        return results

    def fit(self, x_train_dict=None, y_train_dict=None):
        if x_train_dict is None:
            x_train_dict = self.x_train
            y_train_dict = self.y_train
        for category in x_train_dict.keys():
            x_train = x_train_dict[category]
            y_train = y_train_dict[category]
            if self.ws_predictor_prototype is not None:
                self.ws_predictors[category] = deepcopy(self.ws_predictor_prototype)
                self.ws_predictors[category].fit(x_train, y_train)
            
    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump({
                "version": 1,
                "x_train": self.x_train,
                "y_train": self.y_train,
                "ws_predictors": self.ws_predictors,
            }, file)

    def load(self, filename):
        with open(filename, "rb") as file:
            data = pickle.load(file)
            assert data["version"] == 1
            self.x_train = data["x_train"]
            self.y_train = data["y_train"]
            self.ws_predictors = self.ws_predictors

    def _solve(self, model, tee=False):
        if hasattr(self.parent_solver, "set_instance"):
            self.parent_solver.set_instance(model)
            return self.parent_solver.solve(tee=tee, warmstart=True)
        else:
            return self.parent_solver.solve(model, tee=tee, warmstart=True)
