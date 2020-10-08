#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from copy import deepcopy
from typing import Optional, List
from p_tqdm import p_map

from .. import (ObjectiveValueComponent,
                PrimalSolutionComponent,
                DynamicLazyConstraintsComponent,
                UserCutsComponent)
from .pyomo.cplex import CplexPyomoSolver
from .pyomo.gurobi import GurobiPyomoSolver

logger = logging.getLogger(__name__)


# Global memory for multiprocessing
SOLVER = [None]  # type: List[Optional[LearningSolver]]
INSTANCES = [None]  # type: List[Optional[dict]]


def _parallel_solve(instance_idx):
    solver = deepcopy(SOLVER[0])
    instance = INSTANCES[0][instance_idx]
    if not hasattr(instance, "found_violated_lazy_constraints"):
        instance.found_violated_lazy_constraints = []
    if not hasattr(instance, "found_violated_user_cuts"):
        instance.found_violated_user_cuts = []
    if not hasattr(instance, "slacks"):
        instance.slacks = {}
    solver_results = solver.solve(instance)
    return {
        "solver_results": solver_results,
        "solution": instance.solution,
        "lp_solution": instance.lp_solution,
        "found_violated_lazy_constraints": instance.found_violated_lazy_constraints,
        "found_violated_user_cuts": instance.found_violated_user_cuts,
        "slacks": instance.slacks
    }


class LearningSolver:
    def __init__(self,
                 components=None,
                 gap_tolerance=1e-4,
                 mode="exact",
                 solver="gurobi",
                 threads=None,
                 time_limit=None,
                 node_limit=None,
                 solve_lp_first=True,
                 use_lazy_cb=False):
        """
        Mixed-Integer Linear Programming (MIP) solver that extracts information
        from previous runs and uses Machine Learning methods to accelerate the
        solution of new (yet unseen) instances.

        Parameters
        ----------
        components
            Set of components in the solver. By default, includes:
                - ObjectiveValueComponent
                - PrimalSolutionComponent
                - DynamicLazyConstraintsComponent
                - UserCutsComponent
        gap_tolerance
            Relative MIP gap tolerance. By default, 1e-4.
        mode
            If "exact", solves problem to optimality, keeping all optimality
            guarantees provided by the MIP solver. If "heuristic", uses machine
            learning more agressively, and may return suboptimal solutions.
        solver
            The internal MIP solver to use. Can be either "cplex", "gurobi", a
            solver class such as GurobiSolver, or a solver instance such as
            GurobiSolver().
        threads
            Maximum number of threads to use. If None, uses solver default.
        time_limit
            Maximum running time in seconds. If None, uses solver default.
        node_limit
            Maximum number of branch-and-bound nodes to explore. If None, uses
            solver default.
        use_lazy_cb
            If True, uses lazy callbacks to enforce lazy constraints, instead of
            a simple solver loop. This functionality may not supported by
            all internal MIP solvers.
        solve_lp_first: bool
            If true, solve LP relaxation first, then solve original MILP. This
            option should be activated if the LP relaxation is not very
            expensive to solve and if it provides good hints for the integer
            solution.
        """
        self.components = {}
        self.mode = mode
        self.internal_solver = None
        self.internal_solver_factory = solver
        self.threads = threads
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        self.tee = False
        self.node_limit = node_limit
        self.solve_lp_first = solve_lp_first
        self.use_lazy_cb = use_lazy_cb

        if components is not None:
            for comp in components:
                self.add(comp)
        else:
            self.add(ObjectiveValueComponent())
            self.add(PrimalSolutionComponent())
            self.add(DynamicLazyConstraintsComponent())
            self.add(UserCutsComponent())

        assert self.mode in ["exact", "heuristic"]
        for component in self.components.values():
            component.mode = self.mode

    def _create_internal_solver(self):
        logger.debug("Initializing %s" % self.internal_solver_factory)
        if self.internal_solver_factory == "cplex":
            solver = CplexPyomoSolver()
        elif self.internal_solver_factory == "gurobi":
            solver = GurobiPyomoSolver()
        elif callable(self.internal_solver_factory):
            solver = self.internal_solver_factory()
        else:
            solver = self.internal_solver_factory
        if self.threads is not None:
            logger.info("Setting threads to %d" % self.threads)
            solver.set_threads(self.threads)
        if self.time_limit is not None:
            logger.info("Setting time limit to %f" % self.time_limit)
            solver.set_time_limit(self.time_limit)
        if self.gap_tolerance is not None:
            logger.info("Setting gap tolerance to %f" % self.gap_tolerance)
            solver.set_gap_tolerance(self.gap_tolerance)
        if self.node_limit is not None:
            logger.info("Setting node limit to %d" % self.node_limit)
            solver.set_node_limit(self.node_limit)
        return solver

    def solve(self,
              instance,
              model=None,
              tee=False):
        """
        Solves the given instance. If trained machine-learning models are
        available, they will be used to accelerate the solution process.

        This method modifies the instance object. Specifically, the following
        properties are set:
            - instance.lp_solution
            - instance.lp_value
            - instance.lower_bound
            - instance.upper_bound
            - instance.solution
            - instance.solver_log
        Additional solver components may set additional properties. Please
        see their documentation for more details.

        If `solver.solve_lp_first` is False, the properties lp_solution and
        lp_value will be set to dummy values.

        Parameters
        ----------
        instance: miplearn.Instance
            The instance to be solved
        model: pyomo.core.ConcreteModel
            The corresponding Pyomo model. If not provided, it will be created.
        tee: bool
            If true, prints solver log to screen.

        Returns
        -------
        dict
            A dictionary of solver statistics containing at least the following
            keys: "Lower bound", "Upper bound", "Wallclock time", "Nodes",
            "Sense", "Log", "Warm start value" and "LP value".

            Additional components may generate additional keys. For example,
            ObjectiveValueComponent adds the keys "Predicted LB" and
            "Predicted UB". See the documentation of each component for more
            details.
        """

        if model is None:
            model = instance.to_model()

        self.tee = tee
        self.internal_solver = self._create_internal_solver()
        self.internal_solver.set_instance(instance, model)

        if self.solve_lp_first:
            logger.info("Solving LP relaxation...")
            results = self.internal_solver.solve_lp(tee=tee)
            instance.lp_solution = self.internal_solver.get_solution()
            instance.lp_value = results["Optimal value"]
        else:
            instance.lp_solution = self.internal_solver.get_empty_solution()
            instance.lp_value = 0.0

        logger.debug("Running before_solve callbacks...")
        for component in self.components.values():
            component.before_solve(self, instance, model)

        def iteration_cb():
            should_repeat = False
            for comp in self.components.values():
                if comp.after_iteration(self, instance, model):
                    should_repeat = True
            return should_repeat

        def lazy_cb_wrapper(cb_solver, cb_model):
            for comp in self.components.values():
                comp.on_lazy_callback(self, instance, model)

        lazy_cb = None
        if self.use_lazy_cb:
            lazy_cb = lazy_cb_wrapper

        logger.info("Solving MILP...")
        results = self.internal_solver.solve(tee=tee,
                                             iteration_cb=iteration_cb,
                                             lazy_cb=lazy_cb)
        results["LP value"] = instance.lp_value

        # Read MIP solution and bounds
        instance.lower_bound = results["Lower bound"]
        instance.upper_bound = results["Upper bound"]
        instance.solver_log = results["Log"]
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
        self._silence_miplearn_logger()
        SOLVER[0] = self
        INSTANCES[0] = instances
        p_map_results = p_map(_parallel_solve,
                              list(range(len(instances))),
                              num_cpus=n_jobs,
                              desc=label)
        results = [p["solver_results"] for p in p_map_results]
        for (idx, r) in enumerate(p_map_results):
            instances[idx].solution = r["solution"]
            instances[idx].lp_solution = r["lp_solution"]
            instances[idx].lp_value = r["solver_results"]["LP value"]
            instances[idx].lower_bound = r["solver_results"]["Lower bound"]
            instances[idx].upper_bound = r["solver_results"]["Upper bound"]
            instances[idx].found_violated_lazy_constraints = r["found_violated_lazy_constraints"]
            instances[idx].found_violated_user_cuts = r["found_violated_user_cuts"]
            instances[idx].slacks = r["slacks"]
            instances[idx].solver_log = r["solver_results"]["Log"]
        self._restore_miplearn_logger()
        return results

    def fit(self, training_instances):
        if len(training_instances) == 0:
            return
        for component in self.components.values():
            component.fit(training_instances)

    def add(self, component):
        name = component.__class__.__name__
        self.components[name] = component

    def _silence_miplearn_logger(self):
        miplearn_logger = logging.getLogger("miplearn")
        self.prev_log_level = miplearn_logger.getEffectiveLevel()
        miplearn_logger.setLevel(logging.WARNING)    
        
    def _restore_miplearn_logger(self):
        miplearn_logger = logging.getLogger("miplearn")
        miplearn_logger.setLevel(self.prev_log_level)    
        
    def __getstate__(self):
        self.internal_solver = None
        return self.__dict__
