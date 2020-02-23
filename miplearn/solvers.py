#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from . import WarmStartComponent, BranchPriorityComponent
import pyomo.environ as pe
from pyomo.core import Var
from copy import deepcopy
import pickle
from scipy.stats import randint
from p_tqdm import p_map
import logging
logger = logging.getLogger(__name__)


def _solver_factory():
    try:
        solver = pe.SolverFactory('gurobi_persistent')
        assert solver.available()
        solver.options["threads"] = 4
        solver.options["Seed"] = randint(low=0, high=1000).rvs()
        return solver
    except Exception as e:
        logger.debug(e)
        pass

    try:
        solver = pe.SolverFactory('cplex_persistent')
        assert solver.available()
        solver.options["threads"] = 4
        solver.options["randomseed"] = randint(low=0, high=1000).rvs()
        return solver
    except Exception as e:
        logger.debug(e)
        pass

    raise Exception("No solver available")


class LearningSolver:
    """
    Mixed-Integer Linear Programming (MIP) solver that extracts information from previous runs,
    using Machine Learning methods, to accelerate the solution of new (yet unseen) instances.
    """

    def __init__(self,
                 threads=None,
                 time_limit=None,
                 gap_limit=None,
                 internal_solver_factory=_solver_factory,
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
            self.internal_solver.options["timelimit"] = self.time_limit
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
        
        instance.solution = {}
        instance.lower_bound = solve_results["Problem"][0]["Lower bound"]
        instance.upper_bound = solve_results["Problem"][0]["Upper bound"]
        for var in model.component_objects(Var):
            instance.solution[str(var)] = {}
            for index in var:
                instance.solution[str(var)][index] = var[index].value
        
        if self.internal_solver.name == "gurobi_persistent":
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
            return {
                "solver": solver,
                "results": results,
                "solution": instance.solution,
                "upper bound": instance.upper_bound,
                "lower bound": instance.lower_bound,
            }

        p_map_results = p_map(_process, instances, num_cpus=n_jobs, desc=label)
        subsolvers = [p["solver"] for p in p_map_results]
        results = [p["results"] for p in p_map_results]
        
        for (idx, r) in enumerate(p_map_results):
            instances[idx].solution = r["solution"]
            instances[idx].lower_bound = r["lower bound"]
            instances[idx].upper_bound = r["upper bound"]
        
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
