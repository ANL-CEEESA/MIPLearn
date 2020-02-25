#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from . import ObjectiveValueComponent, PrimalSolutionComponent, LazyConstraintsComponent
import pyomo.environ as pe
from pyomo.core import Var
from copy import deepcopy
import pickle
from scipy.stats import randint
from p_tqdm import p_map
import logging
logger = logging.getLogger(__name__)


class InternalSolver:
    def __init__(self):
        self.is_warm_start_available = False
        self.model = None
        pass
    
    def solve_lp(self, tee=False):
        self.solver.set_instance(self.model)
        
        # Relax domain
        from pyomo.core.base.set_types import Reals
        original_domains = []
        for (idx, var) in enumerate(self.model.component_data_objects(Var)):
            original_domains += [var.domain]
            lb, ub = var.bounds
            var.setlb(lb)
            var.setub(ub)
            var.domain = Reals
            self.solver.update_var(var)
        
        # Solve LP relaxation
        results = self.solver.solve(tee=tee)
        
        # Restore domains
        for (idx, var) in enumerate(self.model.component_data_objects(Var)):
            var.domain = original_domains[idx]
            self.solver.update_var(var)
            
        return {
            "Optimal value": results["Problem"][0]["Lower bound"],
        }
            
    def clear_values(self):
        for var in self.model.component_objects(Var):
            for index in var:
                var[index].value = None
                
    def get_solution(self):
        solution = {}
        for var in self.model.component_objects(Var):
            solution[str(var)] = {}
            for index in var:
                solution[str(var)][index] = var[index].value
        return solution   
    
    def set_warm_start(self, ws):
        self.is_warm_start_available = True
        self.clear_values()
        count_total, count_fixed = 0, 0
        for var in ws.keys():
            for index in var:
                count_total += 1
                var[index].value = ws[var][index]
                if ws[var][index] is not None:
                    count_fixed += 1
        logger.info("Setting start values for %d variables (out of %d)" %
                    (count_fixed, count_total))

                
    def set_model(self, model):
        self.model = model
        self.solver.set_instance(model)
        
    def fix(self, ws):
        count_total, count_fixed = 0, 0
        for var in ws.keys():
            for index in var:
                count_total += 1
                if ws[var][index] is None:
                    continue
                count_fixed += 1
                var[index].fix(ws[var][index])
                self.solver.update_var(var[index])        
        logger.info("Fixing values for %d variables (out of %d)" %
                    (count_fixed, count_total))
        
    def add_constraint(self, cut):
        self.solver.add_constraint(cut)

    
class GurobiSolver(InternalSolver):
    def __init__(self):
        super().__init__()
        self.solver = pe.SolverFactory('gurobi_persistent')
        self.solver.options["Seed"] = randint(low=0, high=1000).rvs()
    
    def set_threads(self, threads):
        self.solver.options["Threads"] = threads
    
    def set_time_limit(self, time_limit):
        self.solver.options["TimeLimit"] = time_limit
        
    def set_gap_tolerance(self, gap_tolerance):
        self.solver.options["MIPGap"] = gap_tolerance
        
    def solve(self, tee=False):
        results = self.solver.solve(tee=tee, warmstart=self.is_warm_start_available)
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
        super().__init__()
        import cplex
        self.solver = pe.SolverFactory('cplex_persistent')
        self.solver.options["randomseed"] = randint(low=0, high=1000).rvs()
        
    def set_threads(self, threads):
        self.solver.options["threads"] = threads
    
    def set_time_limit(self, time_limit):
        self.solver.options["timelimit"] = time_limit
        
    def set_gap_tolerance(self, gap_tolerance):
        self.solver.options["mip_tolerances_mipgap"] = gap_tolerance
        
    def solve(self, tee=False):
        results = self.solver.solve(tee=tee, warmstart=self.is_warm_start_available)
        return {
            "Lower bound": results["Problem"][0]["Lower bound"],
            "Upper bound": results["Problem"][0]["Upper bound"],
            "Wallclock time": results["Solver"][0]["Wallclock time"],
            "Nodes": 1,
        }
    
    def solve_lp(self, tee=False):
        import cplex
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
        self.training_instances = []
        
        if self.components is not None:
            assert isinstance(self.components, dict)
        else:
            self.components = {
                "ObjectiveValue": ObjectiveValueComponent(),
                "PrimalSolution": PrimalSolutionComponent(),
                "LazyConstraints": LazyConstraintsComponent(),
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
        self.internal_solver.set_model(model)

        logger.debug("Solving LP relaxation...")
        results = self.internal_solver.solve_lp(tee=tee)
        instance.lp_solution = self.internal_solver.get_solution()
        instance.lp_value = results["Optimal value"]
        
        logger.debug("Running before_solve callbacks...")
        for component in self.components.values():
            component.before_solve(self, instance, model)
        
        if relaxation_only:
            return results
        
        total_wallclock_time = 0
        instance.found_violations = []
        while True:
            logger.debug("Solving MIP...")
            results = self.internal_solver.solve(tee=tee)
            logger.debug("    %.2f s" % results["Wallclock time"])
            total_wallclock_time += results["Wallclock time"]
            if not hasattr(instance, "find_violations"):
                break
            logger.debug("Finding violated constraints...")    
            violations = instance.find_violations(model)
            if len(violations) == 0:
                break
            instance.found_violations += violations
            logger.debug("    %d violations found" % len(violations))
            for v in violations:
                cut = instance.build_lazy_constraint(model, v)
                self.internal_solver.add_constraint(cut)
        results["Wallclock time"] = total_wallclock_time
        
        # Read MIP solution and bounds
        instance.lower_bound = results["Lower bound"]
        instance.upper_bound = results["Upper bound"]
        instance.solution = self.internal_solver.get_solution()
        
        logger.debug("Calling after_solve callbacks...")    
        for component in self.components.values():
            component.after_solve(self, instance, model)
            
        # Store instance for future training
        self.training_instances += [instance]
        
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
                "Violations": instance.found_violations,
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
            instances[idx].found_violations = r["Violations"]
        
        return results

    def fit(self, training_instances=None):
        if training_instances is None:
            training_instances = self.training_instances
        if len(training_instances) == 0:
            return
        for component in self.components.values():
            component.fit(training_instances)
