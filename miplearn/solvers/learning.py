#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import pickle
import os
import tempfile
import gzip

from copy import deepcopy
from typing import Optional, List
from p_tqdm import p_map
from tempfile import NamedTemporaryFile

from . import RedirectOutput
from .. import (
    ObjectiveValueComponent,
    PrimalSolutionComponent,
    DynamicLazyConstraintsComponent,
    UserCutsComponent,
)
from ..solvers.internal import InternalSolver
from ..solvers.pyomo.gurobi import GurobiPyomoSolver

logger = logging.getLogger(__name__)

# Global memory for multiprocessing
SOLVER = [None]  # type: List[Optional[LearningSolver]]
INSTANCES = [None]  # type: List[Optional[dict]]
OUTPUTS = [None]


def _parallel_solve(idx):
    solver = deepcopy(SOLVER[0])
    if OUTPUTS[0] is None:
        output = None
    elif len(OUTPUTS[0]) == 0:
        output = ""
    else:
        output = OUTPUTS[0][idx]
    instance = INSTANCES[0][idx]
    stats = solver.solve(instance, output=output)
    return (stats, instance)


class LearningSolver:
    """
    Mixed-Integer Linear Programming (MIP) solver that extracts information
    from previous runs and uses Machine Learning methods to accelerate the
    solution of new (yet unseen) instances.

    Parameters
    ----------
    components: [Component]
        Set of components in the solver. By default, includes:
            - ObjectiveValueComponent
            - PrimalSolutionComponent
            - DynamicLazyConstraintsComponent
            - UserCutsComponent
    mode: str
        If "exact", solves problem to optimality, keeping all optimality
        guarantees provided by the MIP solver. If "heuristic", uses machine
        learning more aggressively, and may return suboptimal solutions.
    solver: Callable[[], InternalSolver]
        A callable that constructs the internal solver. If None is provided,
        use GurobiPyomoSolver.
    use_lazy_cb: bool
        If true, use native solver callbacks for enforcing lazy constraints,
        instead of a simple loop. May not be supported by all solvers.
    solve_lp_first: bool
        If true, solve LP relaxation first, then solve original MILP. This
        option should be activated if the LP relaxation is not very
        expensive to solve and if it provides good hints for the integer
        solution.
    simulate_perfect: bool
        If true, each call to solve actually performs three actions: solve
        the original problem, train the ML models on the data that was just
        collected, and solve the problem again. This is useful for evaluating
        the theoretical performance of perfect ML models.
    """

    def __init__(
        self,
        components=None,
        mode="exact",
        solver=None,
        use_lazy_cb=False,
        solve_lp_first=True,
        simulate_perfect=False,
    ):
        if solver is None:
            solver = GurobiPyomoSolver
        assert callable(solver), f"Callable expected. Found {solver.__class__} instead."

        self.components = {}
        self.mode = mode
        self.internal_solver = None
        self.solver_factory = solver
        self.use_lazy_cb = use_lazy_cb
        self.tee = False
        self.solve_lp_first = solve_lp_first
        self.simulate_perfect = simulate_perfect

        if components is not None:
            for comp in components:
                self._add_component(comp)
        else:
            self._add_component(ObjectiveValueComponent())
            self._add_component(PrimalSolutionComponent())
            self._add_component(DynamicLazyConstraintsComponent())
            self._add_component(UserCutsComponent())

        assert self.mode in ["exact", "heuristic"]
        for component in self.components.values():
            component.mode = self.mode

    def solve(
        self,
        instance,
        model=None,
        output="",
        tee=False,
    ):
        """
        Solves the given instance. If trained machine-learning models are
        available, they will be used to accelerate the solution process.

        The argument `instance` may be either an Instance object or a
        filename pointing to a pickled Instance object.

        This method modifies the instance object. Specifically, the following
        properties are set:

            - instance.lp_solution
            - instance.lp_value
            - instance.lower_bound
            - instance.upper_bound
            - instance.solution
            - instance.solver_log

        Additional solver components may set additional properties. Please
        see their documentation for more details. If a filename is provided,
        then the file is modified in-place. That is, the original file is
        overwritten.

        If `solver.solve_lp_first` is False, the properties lp_solution and
        lp_value will be set to dummy values.

        Parameters
        ----------
        instance: miplearn.Instance or str
            The instance to be solved, or a filename.
        model: pyomo.core.ConcreteModel
            The corresponding Pyomo model. If not provided, it will be created.
        output: str or None
            If instance is a filename and output is provided, write the modified
            instance to this file, instead of replacing the original file. If
            output is None, discard modified instance.
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
        if self.simulate_perfect:
            if not isinstance(instance, str):
                raise Exception("Not implemented")
            with tempfile.NamedTemporaryFile(suffix=os.path.basename(instance)) as tmp:
                self._solve(
                    instance=instance,
                    model=model,
                    output=tmp.name,
                    tee=tee,
                )
                self.fit([tmp.name])
        return self._solve(
            instance=instance,
            model=model,
            output=output,
            tee=tee,
        )

    def _solve(
        self,
        instance,
        model=None,
        output="",
        tee=False,
    ):
        filename = None
        fileformat = None
        if isinstance(instance, str):
            filename = instance
            logger.info("Reading: %s" % filename)
            if filename.endswith(".gz"):
                fileformat = "pickle-gz"
                with gzip.GzipFile(filename, "rb") as file:
                    instance = pickle.load(file)
            else:
                fileformat = "pickle"
                with open(filename, "rb") as file:
                    instance = pickle.load(file)

        if model is None:
            with RedirectOutput([]):
                model = instance.to_model()

        self.tee = tee
        self.internal_solver = self.solver_factory()
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
                if comp.iteration_cb(self, instance, model):
                    should_repeat = True
            return should_repeat

        def lazy_cb_wrapper(cb_solver, cb_model):
            for comp in self.components.values():
                comp.lazy_cb(self, instance, model)

        lazy_cb = None
        if self.use_lazy_cb:
            lazy_cb = lazy_cb_wrapper

        logger.info("Solving MILP...")
        stats = self.internal_solver.solve(
            tee=tee,
            iteration_cb=iteration_cb,
            lazy_cb=lazy_cb,
        )
        stats["LP value"] = instance.lp_value

        # Read MIP solution and bounds
        instance.lower_bound = stats["Lower bound"]
        instance.upper_bound = stats["Upper bound"]
        instance.solver_log = stats["Log"]
        instance.solution = self.internal_solver.get_solution()

        logger.debug("Calling after_solve callbacks...")
        training_data = {}
        for component in self.components.values():
            component.after_solve(self, instance, model, stats, training_data)

        if not hasattr(instance, "training_data"):
            instance.training_data = []
        instance.training_data += [training_data]

        if filename is not None and output is not None:
            output_filename = output
            if len(output) == 0:
                output_filename = filename
            logger.info("Writing: %s" % output_filename)
            if fileformat == "pickle":
                with open(output_filename, "wb") as file:
                    pickle.dump(instance, file)
            else:
                with gzip.GzipFile(output_filename, "wb") as file:
                    pickle.dump(instance, file)

        return stats

    def parallel_solve(
        self,
        instances,
        n_jobs=4,
        label="Solve",
        output=None,
    ):
        """
        Solves multiple instances in parallel.

        This method is equivalent to calling `solve` for each item on the list,
        but it processes multiple instances at the same time. Like `solve`, this
        method modifies each instance in place. Also like `solve`, a list of
        filenames may be provided.

        Parameters
        ----------
        output: [str] or None
            If instances are file names and output is provided, write the modified
            instances to these files, instead of replacing the original files. If
            output is None, discard modified instance.
        label: str
            Label to show in the progress bar.
        instances: [miplearn.Instance] or [str]
            The instances to be solved
        n_jobs: int
            Number of instances to solve in parallel at a time.

        Returns
        -------
        Returns a list of dictionaries, with one entry for each provided instance.
        This dictionary is the same you would obtain by calling:

            [solver.solve(p) for p in instances]

        """
        if output is None:
            output = []
        self.internal_solver = None
        self._silence_miplearn_logger()
        SOLVER[0] = self
        OUTPUTS[0] = output
        INSTANCES[0] = instances
        results = p_map(
            _parallel_solve,
            list(range(len(instances))),
            num_cpus=n_jobs,
            desc=label,
        )
        stats = []
        for (idx, (s, instance)) in enumerate(results):
            stats.append(s)
            instances[idx] = instance
        self._restore_miplearn_logger()
        return stats

    def fit(self, training_instances):
        if len(training_instances) == 0:
            return
        for component in self.components.values():
            component.fit(training_instances)

    def _add_component(self, component):
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
