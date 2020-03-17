#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from . import ObjectiveValueComponent, PrimalSolutionComponent, LazyConstraintsComponent
import pyomo.environ as pe
from pyomo.core import Var
from copy import deepcopy
from scipy.stats import randint
from p_tqdm import p_map
import logging
logger = logging.getLogger(__name__)


# Global memory for multiprocessing
SOLVER = [None]
INSTANCES = [None]


def _parallel_solve(instance_idx):
    solver = deepcopy(SOLVER[0])
    instance = INSTANCES[0][instance_idx]
    results = solver.solve(instance)
    return {
        "Results": results,
        "Solution": instance.solution,
        "LP solution": instance.lp_solution,
        "LP value": instance.lp_value,
        "Upper bound": instance.upper_bound,
        "Lower bound": instance.lower_bound,
        "Violations": instance.found_violations,
    }


class InternalSolver:
    def __init__(self):
        self.all_vars = None
        self.instance = None
        self.is_warm_start_available = False
        self.model = None
        self.sense = None
        self.solver = None
        self.var_name_to_var = {}

    def solve_lp(self, tee=False):
        self.solver.set_instance(self.model)
        
        # Relax domain
        from pyomo.core.base.set_types import Reals, Binary
        original_domains = []
        for (idx, var) in enumerate(self.model.component_data_objects(Var)):
            original_domains += [var.domain]
            lb, ub = var.bounds
            if var.domain == Binary:
                var.domain = Reals
                var.setlb(lb)
                var.setub(ub)
            self.solver.update_var(var)
        
        # Solve LP relaxation
        results = self.solver.solve(tee=tee)
        
        # Restore domains
        for (idx, var) in enumerate(self.model.component_data_objects(Var)):
            if original_domains[idx] == Binary:
                var.domain = original_domains[idx]
            self.solver.update_var(var)
            
        return {
            "Optimal value": results["Problem"][0]["Lower bound"],
        }
            
    def clear_values(self):
        for var in self.model.component_objects(Var):
            for index in var:
                if var[index].fixed:
                    continue
                var[index].value = None
                
    def get_solution(self):
        solution = {}
        for var in self.model.component_objects(Var):
            solution[str(var)] = {}
            for index in var:
                solution[str(var)][index] = var[index].value
        return solution   
    
    def set_warm_start(self, solution):
        self.is_warm_start_available = True
        self.clear_values()
        count_total, count_fixed = 0, 0
        for var_name in solution:
            var = self.var_name_to_var[var_name]
            for index in solution[var_name]:
                count_total += 1
                var[index].value = solution[var_name][index]
                if solution[var_name][index] is not None:
                    count_fixed += 1
        logger.info("Setting start values for %d variables (out of %d)" %
                    (count_fixed, count_total))
                
    def set_model(self, model):
        from pyomo.core.kernel.objective import minimize
        self.model = model
        self.solver.set_instance(model)
        if self.solver._objective.sense == minimize:
            self.sense = "min"
        else:
            self.sense = "max"
        self.var_name_to_var = {}
        self.all_vars = []
        for var in model.component_objects(Var):
            self.var_name_to_var[var.name] = var
            self.all_vars += [var[idx] for idx in var]
            
    def set_instance(self, instance):
        self.instance = instance
        
    def fix(self, solution):
        count_total, count_fixed = 0, 0
        for var_name in solution:
            for index in solution[var_name]:
                var = self.var_name_to_var[var_name]
                count_total += 1
                if solution[var_name][index] is None:
                    continue
                count_fixed += 1
                var[index].fix(solution[var_name][index])
                self.solver.update_var(var[index])        
        logger.info("Fixing values for %d variables (out of %d)" %
                    (count_fixed, count_total))
    
    def add_constraint(self, cut):
        self.solver.add_constraint(cut)
    
    def solve(self, tee=False):
        total_wallclock_time = 0
        self.instance.found_violations = []
        while True:
            logger.debug("Solving MIP...")
            results = self.solver.solve(tee=tee)
            total_wallclock_time += results["Solver"][0]["Wallclock time"]
            if not hasattr(self.instance, "find_violations"):
                break
            logger.debug("Finding violated constraints...")
            violations = self.instance.find_violations(self.model)
            if len(violations) == 0:
                break
            self.instance.found_violations += violations
            logger.debug("    %d violations found" % len(violations))
            for v in violations:
                cut = self.instance.build_lazy_constraint(self.model, v)
                self.add_constraint(cut)
                
        return {
            "Lower bound": results["Problem"][0]["Lower bound"],
            "Upper bound": results["Problem"][0]["Upper bound"],
            "Wallclock time": total_wallclock_time,
            "Nodes": 1,
            "Sense": self.sense,
        }

    
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
        from gurobipy import GRB

        def cb(cb_model, cb_opt, cb_where):
            if cb_where == GRB.Callback.MIPSOL:
                cb_opt.cbGetSolution(self.all_vars)
                logger.debug("Finding violated constraints...")    
                violations = self.instance.find_violations(cb_model)
                self.instance.found_violations += violations
                logger.debug("    %d violations found" % len(violations))
                for v in violations:
                    cut = self.instance.build_lazy_constraint(cb_model, v)
                    cb_opt.cbLazy(cut)

        if hasattr(self.instance, "find_violations"):
            self.solver.options["LazyConstraints"] = 1
            self.solver.set_callback(cb)
            self.instance.found_violations = []
        results = self.solver.solve(tee=tee, warmstart=self.is_warm_start_available)
        self.solver.set_callback(None)
        return {
            "Lower bound": results["Problem"][0]["Lower bound"],
            "Upper bound": results["Problem"][0]["Upper bound"],
            "Wallclock time": results["Solver"][0]["Wallclock time"],
            "Nodes": self.solver._solver_model.getAttr("NodeCount"),
            "Sense": self.sense,
        }    
            

class CPLEXSolver(InternalSolver):
    def __init__(self):
        super().__init__()
        self.solver = pe.SolverFactory('cplex_persistent')
        self.solver.options["randomseed"] = randint(low=0, high=1000).rvs()
        
    def set_threads(self, threads):
        self.solver.options["threads"] = threads
    
    def set_time_limit(self, time_limit):
        self.solver.options["timelimit"] = time_limit
        
    def set_gap_tolerance(self, gap_tolerance):
        self.solver.options["mip_tolerances_mipgap"] = gap_tolerance
    
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
                 time_limit=None):
        
        self.components = {}
        self.mode = mode
        self.internal_solver = None
        self.internal_solver_factory = solver
        self.threads = threads
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        self.tee = False

        if components is not None:
            for comp in components:
                self.add(comp)
        else:
            self.add(ObjectiveValueComponent())
            self.add(PrimalSolutionComponent())
            self.add(LazyConstraintsComponent())

        assert self.mode in ["exact", "heuristic"]
        for component in self.components.values():
            component.mode = self.mode
        
    def _create_internal_solver(self):
        logger.debug("Initializing %s" % self.internal_solver_factory)
        if self.internal_solver_factory == "cplex":
            solver = CPLEXSolver()
        elif self.internal_solver_factory == "gurobi":
            solver = GurobiSolver()
        else:
            raise Exception("solver %s not supported" % self.internal_solver_factory)
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
              relaxation_only=False):

        if model is None:
            model = instance.to_model()

        self.tee = tee
        self.internal_solver = self._create_internal_solver()
        self.internal_solver.set_model(model)
        self.internal_solver.set_instance(instance)

        logger.debug("Solving LP relaxation...")
        results = self.internal_solver.solve_lp(tee=tee)
        instance.lp_solution = self.internal_solver.get_solution()
        instance.lp_value = results["Optimal value"]
        
        logger.debug("Running before_solve callbacks...")
        for component in self.components.values():
            component.before_solve(self, instance, model)
        
        if relaxation_only:
            return results
        
        results = self.internal_solver.solve(tee=tee)

        # Read MIP solution and bounds
        instance.lower_bound = results["Lower bound"]
        instance.upper_bound = results["Upper bound"]
        instance.solution = self.internal_solver.get_solution()
        
        logger.debug("Calling after_solve callbacks...")    
        for component in self.components.values():
            component.after_solve(self, instance, model, results)

        return results
                
    def parallel_solve(self,
                       instances,
                       n_jobs=4,
                       label="Solve"):
        
        self.internal_solver = None
        SOLVER[0] = self
        INSTANCES[0] = instances
        p_map_results = p_map(_parallel_solve,
                              list(range(len(instances))),
                              num_cpus=n_jobs,
                              desc=label)

        results = [p["Results"] for p in p_map_results]
        for (idx, r) in enumerate(p_map_results):
            instances[idx].solution = r["Solution"]
            instances[idx].lp_solution = r["LP solution"]
            instances[idx].lp_value = r["LP value"]
            instances[idx].lower_bound = r["Lower bound"]
            instances[idx].upper_bound = r["Upper bound"]
            instances[idx].found_violations = r["Violations"]
        
        return results

    def fit(self, training_instances):
        if len(training_instances) == 0:
            return
        for component in self.components.values():
            component.fit(training_instances)

    def add(self, component):
        name = component.__class__.__name__
        self.components[name] = component

    def __getstate__(self):
        self.internal_solver = None
        return self.__dict__
