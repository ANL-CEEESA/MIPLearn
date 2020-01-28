# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from .transformers import PerVariableTransformer
from .warmstart import KnnWarmStartPredictor, LogisticWarmStartPredictor
import pyomo.environ as pe
import numpy as np
from copy import copy, deepcopy
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import randint
import multiprocessing


def _gurobi_factory():
    solver = pe.SolverFactory('gurobi_persistent')
    solver.options["threads"] = 4
    solver.options["Seed"] = randint(low=0, high=1000).rvs()
    return solver

class LearningSolver:
    """
    Mixed-Integer Linear Programming (MIP) solver that extracts information from previous runs,
    using Machine Learning methods, to accelerate the solution of new (yet unseen) instances.
    """

    def __init__(self,
                 threads=4,
                 internal_solver_factory=_gurobi_factory,
                 ws_predictor=LogisticWarmStartPredictor(),
                 branch_priority=None,
                 mode="exact"):
        self.internal_solver_factory = internal_solver_factory
        self.internal_solver = self.internal_solver_factory()
        self.mode = mode
        self.x_train = {}
        self.y_train = {}
        self.ws_predictors = {}
        self.ws_predictor_prototype = ws_predictor
        self.branch_priority = branch_priority

    def solve(self, instance, tee=False):
        # Load model into solver
        model = instance.to_model()
        is_solver_persistent = hasattr(self.internal_solver, "set_instance")
        if is_solver_persistent:
            self.internal_solver.set_instance(model)

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

        for category in var_split.keys():
            var_index_pairs = var_split[category]
            
            # Predict warm starts
            if category in self.ws_predictors.keys():
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

            # Set custom branch priority
            if self.branch_priority is not None:
                assert is_solver_persistent
                from gurobipy import GRB
                for (i, (var, index)) in enumerate(var_index_pairs):
                    gvar = self.internal_solver._pyomo_var_to_solver_var_map[var[index]]
                    #priority = randint(low=0, high=1000).rvs()
                    gvar.setAttr(GRB.Attr.BranchPriority, self.branch_priority[index])

        if is_solver_persistent:
            solve_results = self.internal_solver.solve(tee=tee, warmstart=True)
        else:
            solve_results = self.internal_solver.solve(model, tee=tee, warmstart=True)
            
        solve_results["Solver"][0]["Nodes"] = self.internal_solver._solver_model.getAttr("NodeCount")
            
        
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
        self.parentSolver = None
        
        def _process(instance):
            solver = copy(self)
            solver.internal_solver = solver.internal_solver_factory()
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
            for instance in tqdm(instances, desc=label, ncols=80)
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
