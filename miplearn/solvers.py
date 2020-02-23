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


class InternalSolver():
    def __init__():
        pass
    
    def solve_lp(self, model, tee=False):
        from pyomo.core.base.set_types import Reals
        original_domain = {}
        for var in model.component_data_objects(Var):
            original_domain[str(var)] = var.domain
            lb, ub = var.bounds
            var.setlb(lb)
            var.setub(ub)
            var.domain = Reals
        self.solver.set_instance(model)
        results = self.solver.solve(tee=True)
        for var in model.component_data_objects(Var):
            var.domain = original_domain[str(var)]
        return {
            "Optimal value": results["Problem"][0]["Lower bound"],
        }
            
    def clear_values(self, model):
        for var in model.component_objects(Var):
            for index in var:
                var[index].value = None
                
    def get_solution(self, model):
        solution = {}
        for var in model.component_objects(Var):
            solution[str(var)] = {}
            for index in var:
                solution[str(var)][index] = var[index].value
        return solution        

    
class GurobiSolver(InternalSolver):
    def __init__(self):
        self.solver = pe.SolverFactory('gurobi_persistent')
        self.solver.options["OutputFlag"] = 0
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
            
    def _load_vars(self):
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        vars_to_load = var_map.keys()

        gurobi_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getAttr("X", gurobi_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                var.stale = False
                var.value = val

    
class CPLEXSolver(InternalSolver):
    def __init__(self):
        import cplex
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
        return {
            "Lower bound": results["Problem"][0]["Lower bound"],
            "Upper bound": results["Problem"][0]["Upper bound"],
            "Wallclock time": results["Solver"][0]["Wallclock time"],
            "Nodes": 1,
        }
    
    def solve_lp(self, model, tee=False):
        import cplex
        self.solver.set_instance(model)
        lp = self.solver._solver_model
        var_types = lp.variables.get_types()
        n_vars = len(var_types)
        lp.set_problem_type(cplex.Cplex.problem_type.LP)
        results = self.solver.solve(tee=tee)
        lp.variables.set_types(zip(range(n_vars), var_types))
        return {
            "Optimal value": results["Problem"][0]["Lower bound"],
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
                 solver="gurobi",
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
        
    def solve(self,
              instance,
              model=None,
              tee=False,
              relaxation_only=False,
             ):
        if model is None:
            model = instance.to_model()
            
        self.tee = tee
        self.internal_solver = self._create_internal_solver()

        # Solve LP relaxation
        results = self.internal_solver.solve_lp(model, tee=tee)
        instance.lp_solution = self.internal_solver.get_solution(model)
        instance.lp_value = results["Optimal value"]
        if relaxation_only:
            return results
        
        # Invoke before_solve callbacks
        for component in self.components.values():
            component.before_solve(self, instance, model)
        
        # Check if warm start is available
        is_warm_start_available = False
        if "warm-start" in self.components.keys():
            if self.components["warm-start"].is_warm_start_available:
                is_warm_start_available = True
        
        # Solver original MIP
        self.internal_solver.clear_values(model)
        results = self.internal_solver.solve(model,
                                             tee=tee,
                                             warmstart=is_warm_start_available)
        
        # Read MIP solution and bounds
        instance.lower_bound = results["Lower bound"]
        instance.upper_bound = results["Upper bound"]
        instance.solution = self.internal_solver.get_solution(model)
        
        # Invoke after_solve callbacks
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
                "LP solution": instance.lp_solution,
                "LP value": instance.lp_value,
                "Upper bound": instance.upper_bound,
                "Lower bound": instance.lower_bound,
            }

        p_map_results = p_map(_process, instances, num_cpus=n_jobs, desc=label)
        subsolvers = [p["Solver"] for p in p_map_results]
        results = [p["Results"] for p in p_map_results]
        
        for (idx, r) in enumerate(p_map_results):
            instances[idx].solution = r["Solution"]
            instances[idx].lp_solution = r["LP solution"]
            instances[idx].lp_value = r["LP value"]
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

