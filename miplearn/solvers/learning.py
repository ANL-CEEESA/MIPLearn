#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import gzip
import logging
import os
import pickle
import tempfile
from copy import deepcopy
from typing import Optional, List, Any, IO, cast, BinaryIO, Union, Callable, Dict

from p_tqdm import p_map

from miplearn.components.component import Component
from miplearn.components.cuts import UserCutsComponent
from miplearn.components.lazy_dynamic import DynamicLazyConstraintsComponent
from miplearn.components.objective import ObjectiveValueComponent
from miplearn.components.primal import PrimalSolutionComponent
from miplearn.instance import Instance
from miplearn.solvers import RedirectOutput
from miplearn.solvers.internal import InternalSolver
from miplearn.solvers.pyomo.gurobi import GurobiPyomoSolver
from miplearn.types import MIPSolveStats, TrainingSample

logger = logging.getLogger(__name__)


class GlobalVariables:
    def __init__(self) -> None:
        self.solver: Optional[LearningSolver] = None
        self.instances: Optional[Union[List[str], List[Instance]]] = None
        self.output_filenames: Optional[List[str]] = None
        self.discard_outputs: bool = False


# Global variables used for multiprocessing. Global variables are copied by the
# operating system when the process forks. Local variables are copied through
# serialization, which is a much slower process.
GLOBAL = [GlobalVariables()]


def _parallel_solve(idx):
    solver = GLOBAL[0].solver
    instances = GLOBAL[0].instances
    output_filenames = GLOBAL[0].output_filenames
    discard_outputs = GLOBAL[0].discard_outputs
    if output_filenames is None:
        output_filename = None
    else:
        output_filename = output_filenames[idx]
    stats = solver.solve(
        instances[idx],
        output_filename=output_filename,
        discard_output=discard_outputs,
    )
    return stats, instances[idx]


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
    mode:
        If "exact", solves problem to optimality, keeping all optimality
        guarantees provided by the MIP solver. If "heuristic", uses machine
        learning more aggressively, and may return suboptimal solutions.
    solver:
        A callable that constructs the internal solver. If None is provided,
        use GurobiPyomoSolver.
    use_lazy_cb:
        If true, use native solver callbacks for enforcing lazy constraints,
        instead of a simple loop. May not be supported by all solvers.
    solve_lp_first:
        If true, solve LP relaxation first, then solve original MILP. This
        option should be activated if the LP relaxation is not very
        expensive to solve and if it provides good hints for the integer
        solution.
    simulate_perfect:
        If true, each call to solve actually performs three actions: solve
        the original problem, train the ML models on the data that was just
        collected, and solve the problem again. This is useful for evaluating
        the theoretical performance of perfect ML models.
    """

    def __init__(
        self,
        components: List[Component] = None,
        mode: str = "exact",
        solver: Callable[[], InternalSolver] = None,
        use_lazy_cb: bool = False,
        solve_lp_first: bool = True,
        simulate_perfect: bool = False,
    ):
        if solver is None:
            solver = GurobiPyomoSolver
        assert callable(solver), f"Callable expected. Found {solver.__class__} instead."
        self.components: Dict[str, Component] = {}
        self.internal_solver: Optional[InternalSolver] = None
        self.mode: str = mode
        self.simulate_perfect: bool = simulate_perfect
        self.solve_lp_first: bool = solve_lp_first
        self.solver_factory: Callable[[], InternalSolver] = solver
        self.tee = False
        self.use_lazy_cb: bool = use_lazy_cb
        if components is not None:
            for comp in components:
                self._add_component(comp)
        else:
            self._add_component(ObjectiveValueComponent())
            self._add_component(PrimalSolutionComponent(mode=mode))
            self._add_component(DynamicLazyConstraintsComponent())
            self._add_component(UserCutsComponent())
        assert self.mode in ["exact", "heuristic"]

    def _solve(
        self,
        instance: Union[Instance, str],
        model: Any = None,
        output_filename: Optional[str] = None,
        discard_output: bool = False,
        tee: bool = False,
    ) -> MIPSolveStats:

        # Load instance from file, if necessary
        filename = None
        fileformat = None
        file: Union[BinaryIO, gzip.GzipFile]
        if isinstance(instance, str):
            filename = instance
            logger.info("Reading: %s" % filename)
            if filename.endswith(".gz"):
                fileformat = "pickle-gz"
                with gzip.GzipFile(filename, "rb") as file:
                    instance = pickle.load(cast(IO[bytes], file))
            else:
                fileformat = "pickle"
                with open(filename, "rb") as file:
                    instance = pickle.load(cast(IO[bytes], file))
        assert isinstance(instance, Instance)

        # Generate model
        if model is None:
            with RedirectOutput([]):
                model = instance.to_model()

        # Initialize training sample
        training_sample: TrainingSample = {}
        if not hasattr(instance, "training_data"):
            instance.training_data = []
        instance.training_data += [training_sample]

        # Initialize internal solver
        self.tee = tee
        self.internal_solver = self.solver_factory()
        assert self.internal_solver is not None
        assert isinstance(self.internal_solver, InternalSolver)
        self.internal_solver.set_instance(instance, model)

        # Solve linear relaxation
        if self.solve_lp_first:
            logger.info("Solving LP relaxation...")
            lp_stats = self.internal_solver.solve_lp(tee=tee)
            training_sample["LP solution"] = self.internal_solver.get_solution()
            training_sample["LP value"] = lp_stats["Optimal value"]
            training_sample["LP log"] = lp_stats["Log"]
        else:
            training_sample["LP solution"] = self.internal_solver.get_empty_solution()
            training_sample["LP value"] = 0.0

        # Before-solve callbacks
        logger.debug("Running before_solve callbacks...")
        for component in self.components.values():
            component.before_solve(self, instance, model)

        # Define wrappers
        def iteration_cb_wrapper() -> bool:
            should_repeat = False
            assert isinstance(instance, Instance)
            for comp in self.components.values():
                if comp.iteration_cb(self, instance, model):
                    should_repeat = True
            return should_repeat

        def lazy_cb_wrapper(
            cb_solver: LearningSolver,
            cb_model: Any,
        ) -> None:
            assert isinstance(instance, Instance)
            for comp in self.components.values():
                comp.lazy_cb(self, instance, model)

        lazy_cb = None
        if self.use_lazy_cb:
            lazy_cb = lazy_cb_wrapper

        # Solve MILP
        logger.info("Solving MILP...")
        stats = self.internal_solver.solve(
            tee=tee,
            iteration_cb=iteration_cb_wrapper,
            lazy_cb=lazy_cb,
        )
        if "LP value" in training_sample.keys():
            stats["LP value"] = training_sample["LP value"]

        # Read MIP solution and bounds
        training_sample["Lower bound"] = stats["Lower bound"]
        training_sample["Upper bound"] = stats["Upper bound"]
        training_sample["MIP log"] = stats["Log"]
        training_sample["Solution"] = self.internal_solver.get_solution()

        # After-solve callbacks
        logger.debug("Calling after_solve callbacks...")
        for component in self.components.values():
            component.after_solve(self, instance, model, stats, training_sample)

        # Write to file, if necessary
        if not discard_output and filename is not None:
            if output_filename is None:
                output_filename = filename
            logger.info("Writing: %s" % output_filename)
            if fileformat == "pickle":
                with open(output_filename, "wb") as file:
                    pickle.dump(instance, cast(IO[bytes], file))
            else:
                with gzip.GzipFile(output_filename, "wb") as file:
                    pickle.dump(instance, cast(IO[bytes], file))
        return stats

    def solve(
        self,
        instance: Union[Instance, str],
        model: Any = None,
        output_filename: Optional[str] = None,
        discard_output: bool = False,
        tee: bool = False,
    ) -> MIPSolveStats:
        """
        Solves the given instance. If trained machine-learning models are
        available, they will be used to accelerate the solution process.

        The argument `instance` may be either an Instance object or a
        filename pointing to a pickled Instance object.

        This method adds a new training sample to `instance.training_sample`.
        If a filename is provided, then the file is modified in-place. That is,
        the original file is overwritten.

        If `solver.solve_lp_first` is False, the properties lp_solution and
        lp_value will be set to dummy values.

        Parameters
        ----------
        instance:
            The instance to be solved, or a filename.
        model:
            The corresponding Pyomo model. If not provided, it will be created.
        output_filename:
            If instance is a filename and output_filename is provided, write the
            modified instance to this file, instead of replacing the original one. If
            output_filename is None (the default), modified the original file in-place.
        discard_output:
            If True, do not write the modified instances anywhere; simply discard
            them. Useful during benchmarking.
        tee:
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
                    output_filename=tmp.name,
                    tee=tee,
                )
                self.fit([tmp.name])
        return self._solve(
            instance=instance,
            model=model,
            output_filename=output_filename,
            discard_output=discard_output,
            tee=tee,
        )

    def parallel_solve(
        self,
        instances: Union[List[str], List[Instance]],
        n_jobs: int = 4,
        label: str = "Solve",
        output_filenames: Optional[List[str]] = None,
        discard_outputs: bool = False,
    ) -> List[MIPSolveStats]:
        """
        Solves multiple instances in parallel.

        This method is equivalent to calling `solve` for each item on the list,
        but it processes multiple instances at the same time. Like `solve`, this
        method modifies each instance in place. Also like `solve`, a list of
        filenames may be provided.

        Parameters
        ----------
        output_filenames:
            If instances are file names and output_filenames is provided, write the
            modified instances to these files, instead of replacing the original
            files. If output_filenames is None, modifies the instances in-place.
        discard_outputs:
            If True, do not write the modified instances anywhere; simply discard
            them instead. Useful during benchmarking.
        label:
            Label to show in the progress bar.
        instances:
            The instances to be solved
        n_jobs:
            Number of instances to solve in parallel at a time.

        Returns
        -------
        Returns a list of dictionaries, with one entry for each provided instance.
        This dictionary is the same you would obtain by calling:

            [solver.solve(p) for p in instances]

        """
        self.internal_solver = None
        self._silence_miplearn_logger()
        GLOBAL[0].solver = self
        GLOBAL[0].output_filenames = output_filenames
        GLOBAL[0].instances = instances
        GLOBAL[0].discard_outputs = discard_outputs
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

    def fit(self, training_instances: Union[List[str], List[Instance]]) -> None:
        if len(training_instances) == 0:
            return
        for component in self.components.values():
            component.fit(training_instances)

    def _add_component(self, component: Component) -> None:
        name = component.__class__.__name__
        self.components[name] = component

    def _silence_miplearn_logger(self) -> None:
        miplearn_logger = logging.getLogger("miplearn")
        self.prev_log_level = miplearn_logger.getEffectiveLevel()
        miplearn_logger.setLevel(logging.WARNING)

    def _restore_miplearn_logger(self) -> None:
        miplearn_logger = logging.getLogger("miplearn")
        miplearn_logger.setLevel(self.prev_log_level)

    def __getstate__(self) -> Dict:
        self.internal_solver = None
        return self.__dict__
