#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import sys
from abc import ABC
from copy import deepcopy
from io import StringIO

import pyomo.core.kernel.objective
import pyomo.environ as pe
from p_tqdm import p_map
from pyomo.core import Var
from scipy.stats import randint

from . import (ObjectiveValueComponent,
               PrimalSolutionComponent,
               LazyConstraintsComponent)
from .instance import Instance

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


class RedirectOutput(object):
    def __init__(self, streams):
        self.streams = streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __enter__(self):
        pass

    def __exit__(self, _type, _value, _traceback):
        pass


class InternalSolver(ABC):
    """
    The MIP solver used internaly by LearningSolver.

    Attributes
    ----------
    instance: miplearn.Instance
        The MIPLearn instance currently loaded to the solver
    model: pyomo.core.ConcreteModel
        The Pyomo model currently loaded on the solver
    """

    def __init__(self):
        self.instance = None
        self.model = None
        self._all_vars = None
        self._bin_vars = None
        self._is_warm_start_available = False
        self._pyomo_solver = None
        self._obj_sense = None
        self._varname_to_var = {}

    def solve_lp(self, tee=False):
        """
        Solves the LP relaxation of the currently loaded instance.

        Parameters
        ----------
        tee: bool
            If true, prints the solver log to the screen.

        Returns
        -------
        dict
            A dictionary of solver statistics containing the following keys:
            "Optimal value".
        """
        for var in self._bin_vars:
            lb, ub = var.bounds
            var.setlb(lb)
            var.setub(ub)
            var.domain = pyomo.core.base.set_types.Reals
            self._pyomo_solver.update_var(var)
        results = self._pyomo_solver.solve(tee=tee)
        for var in self._bin_vars:
            var.domain = pyomo.core.base.set_types.Binary
            self._pyomo_solver.update_var(var)
        return {
            "Optimal value": results["Problem"][0]["Lower bound"],
        }
            
    def get_solution(self):
        """
        Returns current solution found by the solver.

        If called after `solve`, returns the best primal solution found during
        the search. If called after `solve_lp`, returns the optimal solution
        to the LP relaxation.

        The solution is a dictionary `sol`, where the optimal value of `var[idx]`
        is given by `sol[var][idx]`.
        """
        solution = {}
        for var in self.model.component_objects(Var):
            solution[str(var)] = {}
            for index in var:
                solution[str(var)][index] = var[index].value
        return solution   
    
    def set_warm_start(self, solution):
        """
        Sets the warm start to be used by the solver.

        The solution should be a dictionary following the same format as the
        one produced by `get_solution`. Only one warm start is currently
        supported. Calling this function when a warm start already exists will
        remove the previous warm start.
        """
        self.clear_warm_start()
        count_total, count_fixed = 0, 0
        for var_name in solution:
            var = self._varname_to_var[var_name]
            for index in solution[var_name]:
                count_total += 1
                var[index].value = solution[var_name][index]
                if solution[var_name][index] is not None:
                    count_fixed += 1
        if count_fixed > 0:
            self._is_warm_start_available = True
        logger.info("Setting start values for %d variables (out of %d)" %
                    (count_fixed, count_total))

    def clear_warm_start(self):
        """
        Removes any existing warm start from the solver.
        """
        for var in self._all_vars:
            if not var.fixed:
                var.value = None
        self._is_warm_start_available = False

    def set_instance(self, instance, model=None):
        """
        Loads the given instance into the solver.

        Parameters
        ----------
        instance: miplearn.Instance
            The instance to be loaded.
        model: pyomo.core.ConcreteModel
            The corresponding Pyomo model. If not provided, it will be
            generated by calling `instance.to_model()`.
        """
        if model is None:
            model = instance.to_model()
        assert isinstance(instance, Instance)
        assert isinstance(model, pe.ConcreteModel)
        self.instance = instance
        self.model = model
        self._pyomo_solver.set_instance(model)

        # Update objective sense
        self._obj_sense = "max"
        if self._pyomo_solver._objective.sense == pyomo.core.kernel.objective.minimize:
            self._obj_sense = "min"

        # Update variables
        self._all_vars = []
        self._bin_vars = []
        self._varname_to_var = {}
        for var in model.component_objects(Var):
            self._varname_to_var[var.name] = var
            for idx in var:
                self._all_vars += [var[idx]]
                if var[idx].domain == pyomo.core.base.set_types.Binary:
                    self._bin_vars += [var[idx]]

    def fix(self, solution):
        """
        Fixes the values of a subset of decision variables.

        The values should be provided in the dictionary format generated by
        `get_solution`. Missing values in the solution indicate variables
        that should be left free.
        """
        count_total, count_fixed = 0, 0
        for varname in solution:
            for index in solution[varname]:
                var = self._varname_to_var[varname]
                count_total += 1
                if solution[varname][index] is None:
                    continue
                count_fixed += 1
                var[index].fix(solution[varname][index])
                self._pyomo_solver.update_var(var[index])
        logger.info("Fixing values for %d variables (out of %d)" %
                    (count_fixed, count_total))
    
    def add_constraint(self, constraint):
        """
        Adds a single constraint to the model.
        """
        self._pyomo_solver.add_constraint(constraint)

    def solve(self, tee=False):
        """
        Solves the currently loaded instance.

        Parameters
        ----------
        tee: bool
            If true, prints the solver log to the screen.

        Returns
        -------
        dict
            A dictionary of solver statistics containing the following keys:
            "Lower bound", "Upper bound", "Wallclock time", "Nodes", "Sense"
            and "Log".
        """
        total_wallclock_time = 0
        streams = [StringIO()]
        if tee:
            streams += [sys.stdout]
        self.instance.found_violations = []
        while True:
            logger.debug("Solving MIP...")
            with RedirectOutput(streams):
                results = self._pyomo_solver.solve(tee=True)
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
            "Sense": self._obj_sense,
            "Log": streams[0].getvalue()
        }


class GurobiSolver(InternalSolver):
    def __init__(self):
        super().__init__()
        self._pyomo_solver = pe.SolverFactory('gurobi_persistent')
        self._pyomo_solver.options["Seed"] = randint(low=0, high=1000).rvs()
    
    def set_threads(self, threads):
        self._pyomo_solver.options["Threads"] = threads
    
    def set_time_limit(self, time_limit):
        self._pyomo_solver.options["TimeLimit"] = time_limit
        
    def set_gap_tolerance(self, gap_tolerance):
        self._pyomo_solver.options["MIPGap"] = gap_tolerance
        
    def solve(self, tee=False):
        from gurobipy import GRB

        def cb(cb_model, cb_opt, cb_where):
            if cb_where == GRB.Callback.MIPSOL:
                cb_opt.cbGetSolution(self._all_vars)
                logger.debug("Finding violated constraints...")    
                violations = self.instance.find_violations(cb_model)
                self.instance.found_violations += violations
                logger.debug("    %d violations found" % len(violations))
                for v in violations:
                    cut = self.instance.build_lazy_constraint(cb_model, v)
                    cb_opt.cbLazy(cut)

        if hasattr(self.instance, "find_violations"):
            self._pyomo_solver.options["LazyConstraints"] = 1
            self._pyomo_solver.set_callback(cb)
            self.instance.found_violations = []
        print(self._is_warm_start_available)

        streams = [StringIO()]
        if tee:
            streams += [sys.stdout]
        with RedirectOutput(streams):
            results = self._pyomo_solver.solve(tee=True,
                                               warmstart=self._is_warm_start_available)
        self._pyomo_solver.set_callback(None)
        node_count = int(self._pyomo_solver._solver_model.getAttr("NodeCount"))
        return {
            "Lower bound": results["Problem"][0]["Lower bound"],
            "Upper bound": results["Problem"][0]["Upper bound"],
            "Wallclock time": results["Solver"][0]["Wallclock time"],
            "Nodes": max(1, node_count),
            "Sense": self._obj_sense,
            "Log": streams[0].getvalue(),
        }    
            

class CPLEXSolver(InternalSolver):
    def __init__(self,
                 presolve=1,
                 mip_display=4,
                 threads=None,
                 time_limit=None,
                 gap_tolerance=None):
        super().__init__()
        self._pyomo_solver = pe.SolverFactory('cplex_persistent')
        self._pyomo_solver.options["randomseed"] = randint(low=0, high=1000).rvs()
        self._pyomo_solver.options["preprocessing_presolve"] = presolve
        self._pyomo_solver.options["mip_display"] = mip_display
        if threads is not None:
            self.set_threads(threads)
        if time_limit is not None:
            self.set_time_limit(time_limit)
        if gap_tolerance is not None:
            self.set_gap_tolerance(gap_tolerance)
        
    def set_threads(self, threads):
        self._pyomo_solver.options["threads"] = threads
    
    def set_time_limit(self, time_limit):
        self._pyomo_solver.options["timelimit"] = time_limit
        
    def set_gap_tolerance(self, gap_tolerance):
        self._pyomo_solver.options["mip_tolerances_mipgap"] = gap_tolerance
    
    def solve_lp(self, tee=False):
        import cplex
        lp = self._pyomo_solver._solver_model
        var_types = lp.variables.get_types()
        n_vars = len(var_types)
        lp.set_problem_type(cplex.Cplex.problem_type.LP)
        results = self._pyomo_solver.solve(tee=tee)
        lp.variables.set_types(zip(range(n_vars), var_types))
        return {
            "Optimal value": results["Problem"][0]["Lower bound"],
        }
        

class LearningSolver:
    """
    Mixed-Integer Linear Programming (MIP) solver that extracts information
    from previous runs, using Machine Learning methods, to accelerate the
    solution of new (yet unseen) instances.
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
        elif callable(self.internal_solver_factory):
            solver = self.internal_solver_factory()
            assert isinstance(solver, InternalSolver)
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
        self.internal_solver.set_instance(instance, model=model)

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
