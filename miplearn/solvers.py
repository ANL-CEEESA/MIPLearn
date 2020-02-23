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


class GurobiSolver:
    def __init__(self):
        self.solver = pe.SolverFactory('gurobi_persistent')
        self.solver.options["Seed"] = randint(low=0, high=1000).rvs()
    
    def set_threads(self, threads):
        self.solver.options["Threads"] = threads
    
    def set_time_limit(self, time_limit):
        self.solver.options["TimeLimit"] = time_limit
        
    def set_gap_tolerance(self, gap_tolerance):
        self.solver.options["MIPGap"] = gap_tolerance
        
    def solve(self, model, tee=False, warmstart=False):
        self.solver.set_instance(model)
        results = self.solver.solve(tee=tee, warmstart=warmstart)
        return {
            "Lower bound": results["Problem"][0]["Lower bound"],
            "Upper bound": results["Problem"][0]["Upper bound"],
            "Wallclock time": results["Solver"][0]["Wallclock time"],
            "Nodes": self.solver._solver_model.getAttr("NodeCount"),
        }    

    
class CPLEXSolver:
    def __init__(self):
        self.solver = pe.SolverFactory('cplex_persistent')
        self.solver.options["randomseed"] = randint(low=0, high=1000).rvs()
        
    def set_threads(self, threads):
        self.solver.options["threads"] = threads
    
    def set_time_limit(self, time_limit):
        self.solver.options["timelimit"] = time_limit
        
    def set_gap_tolerance(self, gap_tolerance):
        self.solver.options["mip_tolerances_mipgap"] = gap_tolerance
        
    def solve(self, model, tee=False, warmstart=False):
        self.solver.set_instance(model)
        results = self.solver.solve(tee=tee, warmstart=warmstart)
        print(results)
        return {
            "Lower bound": results["Problem"][0]["Lower bound"],
            "Upper bound": results["Problem"][0]["Upper bound"],
            "Wallclock time": results["Solver"][0]["Wallclock time"],
            "Nodes": 1,
        }


class LearningSolver:
    """
    Mixed-Integer Linear Programming (MIP) solver that extracts information from previous runs,
    using Machine Learning methods, to accelerate the solution of new (yet unseen) instances.
    """

    def __init__(self,
                 components=None,
                 gap_tolerance=None,
                 mode="exact",
                 solver="cplex",
                 threads=4,
                 time_limit=None,
                ):
        
        self.is_persistent = None
        self.components = components
        self.mode = mode
        self.internal_solver = None
        self.internal_solver_factory = solver
        self.threads = threads
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        self.tee = False
        
        if self.components is not None:
            assert isinstance(self.components, dict)
        else:
            self.components = {
                "warm-start": WarmStartComponent(),
            }
            
        assert self.mode in ["exact", "heuristic"]
        for component in self.components.values():
            component.mode = self.mode
        
    def _create_internal_solver(self):
        if self.internal_solver_factory == "cplex":
            solver = CPLEXSolver()
        elif self.internal_solver_factory == "gurobi":
            solver = GurobiSolver()
        else:
            raise Exception("solver %s not supported" % solver_factory)
        solver.set_threads(self.threads)
        if self.time_limit is not None:
            solver.set_time_limit(self.time_limit)
        if self.gap_tolerance is not None:
            solver.set_gap_tolerance(self.gap_tolerance)
        return solver
        
    def solve(self, instance, tee=False):
        model = instance.to_model()
        self.tee = tee

        self.internal_solver = self._create_internal_solver()
        
        for component in self.components.values():
            component.before_solve(self, instance, model)
        
        is_warm_start_available = False
        if "warm-start" in self.components.keys():
            if self.components["warm-start"].is_warm_start_available:
                is_warm_start_available = True
        
        results = self.internal_solver.solve(model,
                                                   tee=tee,
                                                   warmstart=is_warm_start_available)
        
        instance.solution = {}
        instance.lower_bound = results["Lower bound"]
        instance.upper_bound = results["Upper bound"]
        
        for var in model.component_objects(Var):
            instance.solution[str(var)] = {}
            for index in var:
                instance.solution[str(var)][index] = var[index].value
        
        for component in self.components.values():
            component.after_solve(self, instance, model)
        
        return results
                
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
                "Solver": solver,
                "Results": results,
                "Solution": instance.solution,
                "Upper bound": instance.upper_bound,
                "Lower bound": instance.lower_bound,
            }

        p_map_results = p_map(_process, instances, num_cpus=n_jobs, desc=label)
        subsolvers = [p["Solver"] for p in p_map_results]
        results = [p["Results"] for p in p_map_results]
        
        for (idx, r) in enumerate(p_map_results):
            instances[idx].solution = r["Solution"]
            instances[idx].lower_bound = r["Lower bound"]
            instances[idx].upper_bound = r["Upper bound"]
        
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
