# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from .transformers import PerVariableTransformer
from .warmstart import WarmStartComponent
from .branching import BranchPriorityComponent
import pyomo.environ as pe
import numpy as np
from copy import deepcopy
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
                 components=None,
                 mode=None):
        self.is_persistent = None
        self.internal_solver = None
        self.components = components
        self.internal_solver_factory = internal_solver_factory
        
        if self.components is not None:
            assert isinstance(self.components, dict)
        else:
            self.components = {
                "warm-start": WarmStartComponent(),
                #"branch-priority": BranchPriorityComponent(),
            }
            
        if mode is not None:
            assert mode in ["exact", "heuristic"]
            for component in self.components.values():
                component.mode = mode
        
    def _create_solver(self):
        self.internal_solver = self.internal_solver_factory()
        self.is_persistent = hasattr(self.internal_solver, "set_instance")
        
    def _clear(self):
        self.internal_solver = None

    def solve(self, instance, tee=False):
        model = instance.to_model()
        
        self._create_solver()
        if self.is_persistent:
            self.internal_solver.set_instance(model)
        
        for component in self.components.values():
            component.before_solve(self, instance, model)
        
        if self.is_persistent:
            solve_results = self.internal_solver.solve(tee=tee, warmstart=True)
        else:
            solve_results = self.internal_solver.solve(model, tee=tee, warmstart=True)
        
        solve_results["Solver"][0]["Nodes"] = self.internal_solver._solver_model.getAttr("NodeCount")
        
        for component in self.components.values():
            component.after_solve(self, instance, model)
        
        return solve_results
                
    def parallel_solve(self, instances, n_jobs=4, label="Solve"):
        self._clear()
        
        def _process(instance):
            solver = deepcopy(self)
            results = solver.solve(instance)
            solver._clear()
            return solver, results

        solver_result_pairs = Parallel(n_jobs=n_jobs)(
            delayed(_process)(instance)
            for instance in tqdm(instances, desc=label, ncols=80)
        )
        
        solvers = [p[0] for p in solver_result_pairs]
        results = [p[1] for p in solver_result_pairs]
        
        for (name, component) in self.components.items():
            for subsolver in solvers:
                self.components[name].merge(subsolver.components[name])
        
        return results

    def fit(self):
        for component in self.components.values():
            component.fit(self)
            
    def save_state(self, filename):
        with open(filename, "wb") as file:
            pickle.dump({
                "version": 2,
                "components": self.components,
            }, file)

    def load_state(self, filename):
        with open(filename, "rb") as file:
            data = pickle.load(file)
            assert data["version"] == 2
            for (component_name, component) in data["components"].items():
                if component_name not in self.components.keys():
                    continue
                else:
                    self.components[component_name].merge(component)
