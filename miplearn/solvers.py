# MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from .warmstart import *
import pyomo.environ as pe
import numpy as np
from math import isfinite

class LearningSolver:
    """
    Mixed-Integer Linear Programming (MIP) solver that extracts information from previous runs,
    using Machine Learning methods, to accelerate the solution of new (yet unseen) instances.
    """
    
    def __init__(self,
                 threads = 4,
                 ws_predictor = None):
        self.parent_solver = pe.SolverFactory('cplex_persistent')
        self.parent_solver.options["threads"] = threads
        self.train_x = None
        self.train_y = None
        self.ws_predictor = ws_predictor
        
    def solve(self,
              instance,
              tee=False,
              learn=True):
        model = instance.to_model()
        self.parent_solver.set_instance(model)
        self.cplex = self.parent_solver._solver_model
        x = self._get_features(instance)
        
        if self.ws_predictor is not None:
            self.cplex.MIP_starts.delete()
            ws = self.ws_predictor.predict(x)
            if ws is not None:
                _add_warm_start(self.cplex, ws)
        
        self.parent_solver.solve(tee=tee)

        solution = np.array(self.cplex.solution.get_values())
        y = np.transpose(np.vstack((solution, 1 - solution)))
        self._update_training_set(x, y)
        return y
    
    def transform(self, instance):
        model = instance.to_model()
        self.parent_solver.set_instance(model)
        self.cplex = self.parent_solver._solver_model
        return self._get_features(instance)
    
    def predict(self, instance):
        pass

    def _update_training_set(self, x, y):
        if self.train_x is None:
            self.train_x = x
            self.train_y = y
        else:
            self.train_x = np.vstack((self.train_x, x))
            self.train_y = np.vstack((self.train_y, y))
        
    def fit(self):
        if self.ws_predictor is not None:
            self.ws_predictor.fit(self.train_x, self.train_y)

def _add_warm_start(cplex, ws):
    assert isinstance(ws, np.ndarray)
    assert ws.shape == (cplex.variables.get_num(),)
    indices, values = [], []
    for k in range(len(ws)):
        if isfinite(ws[k]):
            indices += [k]
            values += [ws[k]]
    print("Adding warm start with %d values" % len(indices))
    cplex.MIP_starts.add([indices, values], cplex.MIP_starts.effort_level.solve_MIP)

