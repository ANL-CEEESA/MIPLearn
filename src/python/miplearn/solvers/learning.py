#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from copy import deepcopy
from typing import Optional, List

from p_tqdm import p_map

from .cplex import CPLEXSolver
from .gurobi import GurobiSolver
from .internal import InternalSolver
from .. import (ObjectiveValueComponent,
                PrimalSolutionComponent,
                LazyConstraintsComponent)

logger = logging.getLogger(__name__)


# Global memory for multiprocessing
SOLVER = [None]  # type: List[Optional[LearningSolver]]
INSTANCES = [None]  # type: List[Optional[dict]]


def _parallel_solve(instance_idx):
    solver = deepcopy(SOLVER[0])
    instance = INSTANCES[0][instance_idx]
    results = solver.solve(instance)
    return {
        "Results": results,
        "Solution": instance.solution,
        "LP solution": instance.lp_solution,
        "Violations": instance.found_violations,
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
            - instance.found_violations
            - instance.solver_log
        Additional solver components may set additional properties. Please
        see their documentation for more details.

        Parameters
        ----------
        instance: miplearn.Instance
            The instance to be solved
        model: pyomo.core.ConcreteModel
            The corresponding Pyomo model. If not provided, it will be created.
        tee: bool
            If true, prints solver log to screen.
        relaxation_only: bool
            If true, solve only the root LP relaxation.

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
            instances[idx].lp_value = r["Results"]["LP value"]
            instances[idx].lower_bound = r["Results"]["Lower bound"]
            instances[idx].upper_bound = r["Results"]["Upper bound"]
            instances[idx].found_violations = r["Violations"]
            instances[idx].solver_log = r["Results"]["Log"]

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
