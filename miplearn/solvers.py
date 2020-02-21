# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright © 2020, UChicago Argonne, LLC. All rights reserved.
# Released under the modified BSD license. See COPYING.md for more details.
# Written by Alinson S. Xavier <axavier@anl.gov>

from . import WarmStartComponent, BranchPriorityComponent
import pyomo.environ as pe
import numpy as np
from copy import deepcopy
import pickle
from scipy.stats import randint
from p_tqdm import p_map

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
                 threads=None,
                 time_limit=None,
                 gap_limit=None,
                 internal_solver_factory=_gurobi_factory,
                 components=None,
                 mode="exact"):
        self.is_persistent = None
        self.internal_solver = None
        self.components = components
        self.internal_solver_factory = internal_solver_factory
        self.threads = threads
        self.time_limit = time_limit
        self.gap_limit = gap_limit
        self.tee = False
        self.mode = mode
        
        if self.components is not None:
            assert isinstance(self.components, dict)
        else:
            self.components = {
                "warm-start": WarmStartComponent(),
            }
            
        assert self.mode in ["exact", "heuristic"]
        for component in self.components.values():
            component.mode = self.mode
        
    def _create_solver(self):
        self.internal_solver = self.internal_solver_factory()
        self.is_persistent = hasattr(self.internal_solver, "set_instance")
        if self.threads is not None:
            self.internal_solver.options["Threads"] = self.threads
        if self.time_limit is not None:
            self.internal_solver.options["TimeLimit"] = self.time_limit
        if self.gap_limit is not None:
            self.internal_solver.options["MIPGap"] = self.gap_limit
        
    def solve(self, instance, tee=False):
        model = instance.to_model()
        self.tee = tee

        self._create_solver()
        if self.is_persistent:
            self.internal_solver.set_instance(model)
        
        for component in self.components.values():
            component.before_solve(self, instance, model)
        
        is_warm_start_available = False
        if "warm-start" in self.components.keys():
            if self.components["warm-start"].is_warm_start_available:
                is_warm_start_available = True
        if self.is_persistent:
            solve_results = self.internal_solver.solve(tee=tee, warmstart=is_warm_start_available)
        else:
            solve_results = self.internal_solver.solve(model, tee=tee, warmstart=is_warm_start_available)
        
        if hasattr(self.internal_solver, "_solver_model"):
            solve_results["Solver"][0]["Nodes"] = self.internal_solver._solver_model.getAttr("NodeCount")
        else:
            solve_results["Solver"][0]["Nodes"] = 1
        
        for component in self.components.values():
            component.after_solve(self, instance, model)
        
        return solve_results
                
    def parallel_solve(self,
                       instances,
                       n_jobs=4,
                       label="Solve",
                       collect_training_data=True,
                      ):
        self.internal_solver = None
        
        def _process(instance):
            solver = deepcopy(self)
            results = solver.solve(instance)
            solver.internal_solver = None
            if not collect_training_data:
                solver.components = {}
            return solver, results

        solver_result_pairs = p_map(_process, instances, num_cpus=n_jobs, desc=label)
        subsolvers = [p[0] for p in solver_result_pairs]
        results = [p[1] for p in solver_result_pairs]
        
        for (name, component) in self.components.items():
            subcomponents = [subsolver.components[name]
                             for subsolver in subsolvers
                             if name in subsolver.components.keys()]
            self.components[name].merge(subcomponents)
        
        return results

    def fit(self, n_jobs=1):
        for component in self.components.values():
            component.fit(self, n_jobs=n_jobs)
            
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
                    self.components[component_name].merge([component])
