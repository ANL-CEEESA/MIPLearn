#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import traceback
from typing import Optional, List, Any, cast, Dict, Tuple

from p_tqdm import p_map

from miplearn.components.component import Component
from miplearn.components.dynamic_lazy import DynamicLazyConstraintsComponent
from miplearn.components.dynamic_user_cuts import UserCutsComponent
from miplearn.components.objective import ObjectiveValueComponent
from miplearn.components.primal import PrimalSolutionComponent
from miplearn.features import FeaturesExtractor, Sample
from miplearn.instance.base import Instance
from miplearn.instance.picklegz import PickleGzInstance
from miplearn.solvers import _RedirectOutput
from miplearn.solvers.internal import InternalSolver
from miplearn.solvers.pyomo.gurobi import GurobiPyomoSolver
from miplearn.types import LearningSolveStats

logger = logging.getLogger(__name__)


class _GlobalVariables:
    def __init__(self) -> None:
        self.solver: Optional[LearningSolver] = None
        self.instances: Optional[List[Instance]] = None
        self.discard_outputs: bool = False


# Global variables used for multiprocessing. Global variables are copied by the
# operating system when the process forks. Local variables are copied through
# serialization, which is a much slower process.
_GLOBAL = [_GlobalVariables()]


def _parallel_solve(
    idx: int,
) -> Tuple[Optional[LearningSolveStats], Optional[Instance]]:
    solver = _GLOBAL[0].solver
    instances = _GLOBAL[0].instances
    discard_outputs = _GLOBAL[0].discard_outputs
    assert solver is not None
    assert instances is not None
    try:
        stats = solver.solve(
            instances[idx],
            discard_output=discard_outputs,
        )
        instances[idx].free()
        return stats, instances[idx]
    except Exception as e:
        traceback.print_exc()
        logger.exception(f"Exception while solving {instances[idx]}. Ignoring.")
        return None, None


class LearningSolver:
    """
    Mixed-Integer Linear Programming (MIP) solver that extracts information
    from previous runs and uses Machine Learning methods to accelerate the
    solution of new (yet unseen) instances.

    Parameters
    ----------
    components: List[Component]
        Set of components in the solver. By default, includes
        `ObjectiveValueComponent`, `PrimalSolutionComponent`,
        `DynamicLazyConstraintsComponent` and `UserCutsComponent`.
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
    solve_lp: bool
        If true, solve the root LP relaxation before solving the MIP. This
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
        components: Optional[List[Component]] = None,
        mode: str = "exact",
        solver: Optional[InternalSolver] = None,
        use_lazy_cb: bool = False,
        solve_lp: bool = True,
        simulate_perfect: bool = False,
        extractor: Optional[FeaturesExtractor] = None,
    ) -> None:
        if solver is None:
            solver = GurobiPyomoSolver()
        if extractor is None:
            extractor = FeaturesExtractor()
        assert isinstance(solver, InternalSolver)
        self.components: Dict[str, Component] = {}
        self.internal_solver: Optional[InternalSolver] = None
        self.internal_solver_prototype: InternalSolver = solver
        self.mode: str = mode
        self.simulate_perfect: bool = simulate_perfect
        self.solve_lp: bool = solve_lp
        self.tee = False
        self.use_lazy_cb: bool = use_lazy_cb
        self.extractor = extractor
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
        instance: Instance,
        model: Any = None,
        discard_output: bool = False,
        tee: bool = False,
    ) -> LearningSolveStats:

        # Generate model
        # -------------------------------------------------------
        instance.load()
        if model is None:
            with _RedirectOutput([]):
                model = instance.to_model()

        # Initialize training sample
        # -------------------------------------------------------
        sample = Sample()
        instance.samples.append(sample)

        # Initialize stats
        # -------------------------------------------------------
        stats: LearningSolveStats = {}

        # Initialize internal solver
        # -------------------------------------------------------
        self.tee = tee
        self.internal_solver = self.internal_solver_prototype.clone()
        assert self.internal_solver is not None
        assert isinstance(self.internal_solver, InternalSolver)
        self.internal_solver.set_instance(instance, model)

        # Extract features (after-load)
        # -------------------------------------------------------
        logger.info("Extracting features (after-load)...")
        features = self.extractor.extract(instance, self.internal_solver)
        features.extra = {}
        sample.after_load = features

        callback_args = (
            self,
            instance,
            model,
            stats,
            sample,
        )

        # Solve root LP relaxation
        # -------------------------------------------------------
        lp_stats = None
        if self.solve_lp:
            logger.debug("Running before_solve_lp callbacks...")
            for component in self.components.values():
                component.before_solve_lp(*callback_args)

            logger.info("Solving root LP relaxation...")
            lp_stats = self.internal_solver.solve_lp(tee=tee)
            stats.update(cast(LearningSolveStats, lp_stats.__dict__))

            logger.debug("Running after_solve_lp callbacks...")
            for component in self.components.values():
                component.after_solve_lp(*callback_args)

            # Extract features (after-lp)
            # -------------------------------------------------------
            logger.info("Extracting features (after-lp)...")
            features = self.extractor.extract(
                instance,
                self.internal_solver,
                with_static=False,
            )
            features.extra = {}
            features.lp_solve = lp_stats
            sample.after_lp = features

        # Callback wrappers
        # -------------------------------------------------------
        def iteration_cb_wrapper() -> bool:
            should_repeat = False
            for comp in self.components.values():
                if comp.iteration_cb(self, instance, model):
                    should_repeat = True
            return should_repeat

        def lazy_cb_wrapper(
            cb_solver: InternalSolver,
            cb_model: Any,
        ) -> None:
            for comp in self.components.values():
                comp.lazy_cb(self, instance, model)

        def user_cut_cb_wrapper(
            cb_solver: InternalSolver,
            cb_model: Any,
        ) -> None:
            for comp in self.components.values():
                comp.user_cut_cb(self, instance, model)

        lazy_cb = None
        if self.use_lazy_cb:
            lazy_cb = lazy_cb_wrapper

        user_cut_cb = None
        if instance.has_user_cuts():
            user_cut_cb = user_cut_cb_wrapper

        # Before-solve callbacks
        # -------------------------------------------------------
        logger.debug("Running before_solve_mip callbacks...")
        for component in self.components.values():
            component.before_solve_mip(*callback_args)

        # Solve MIP
        # -------------------------------------------------------
        logger.info("Solving MIP...")
        mip_stats = self.internal_solver.solve(
            tee=tee,
            iteration_cb=iteration_cb_wrapper,
            user_cut_cb=user_cut_cb,
            lazy_cb=lazy_cb,
        )
        stats.update(cast(LearningSolveStats, mip_stats.__dict__))
        stats["Solver"] = "default"
        stats["Gap"] = self._compute_gap(
            ub=mip_stats.mip_upper_bound,
            lb=mip_stats.mip_lower_bound,
        )
        stats["Mode"] = self.mode

        # Extract features (after-mip)
        # -------------------------------------------------------
        logger.info("Extracting features (after-mip)...")
        features = self.extractor.extract(
            instance,
            self.internal_solver,
            with_static=False,
        )
        features.mip_solve = mip_stats
        features.extra = {}
        sample.after_mip = features

        # After-solve callbacks
        # -------------------------------------------------------
        logger.debug("Calling after_solve_mip callbacks...")
        for component in self.components.values():
            component.after_solve_mip(*callback_args)

        # Flush
        # -------------------------------------------------------
        if not discard_output:
            instance.flush()

        return stats

    def solve(
        self,
        instance: Instance,
        model: Any = None,
        discard_output: bool = False,
        tee: bool = False,
    ) -> LearningSolveStats:
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
        instance: Instance
            The instance to be solved.
        model: Any
            The corresponding Pyomo model. If not provided, it will be created.
        discard_output: bool
            If True, do not write the modified instances anywhere; simply discard
            them. Useful during benchmarking.
        tee: bool
            If true, prints solver log to screen.

        Returns
        -------
        LearningSolveStats
            A dictionary of solver statistics containing at least the following
            keys: "Lower bound", "Upper bound", "Wallclock time", "Nodes",
            "Sense", "Log", "Warm start value" and "LP value".

            Additional components may generate additional keys. For example,
            ObjectiveValueComponent adds the keys "Predicted LB" and
            "Predicted UB". See the documentation of each component for more
            details.
        """
        if self.simulate_perfect:
            if not isinstance(instance, PickleGzInstance):
                raise Exception("Not implemented")
            self._solve(
                instance=instance,
                model=model,
                tee=tee,
            )
            self.fit([instance])
            instance.instance = None
        return self._solve(
            instance=instance,
            model=model,
            discard_output=discard_output,
            tee=tee,
        )

    def parallel_solve(
        self,
        instances: List[Instance],
        n_jobs: int = 4,
        label: str = "Solve",
        discard_outputs: bool = False,
    ) -> List[LearningSolveStats]:
        """
        Solves multiple instances in parallel.

        This method is equivalent to calling `solve` for each item on the list,
        but it processes multiple instances at the same time. Like `solve`, this
        method modifies each instance in place. Also like `solve`, a list of
        filenames may be provided.

        Parameters
        ----------
        discard_outputs: bool
            If True, do not write the modified instances anywhere; simply discard
            them instead. Useful during benchmarking.
        label: str
            Label to show in the progress bar.
        instances: List[Instance]
            The instances to be solved.
        n_jobs: int
            Number of instances to solve in parallel at a time.

        Returns
        -------
        List[LearningSolveStats]
            List of solver statistics, with one entry for each provided instance.
            The list is the same you would obtain by calling
            `[solver.solve(p) for p in instances]`
        """
        if n_jobs == 1:
            return [self.solve(p) for p in instances]
        else:
            self.internal_solver = None
            self._silence_miplearn_logger()
            _GLOBAL[0].solver = self
            _GLOBAL[0].instances = instances
            _GLOBAL[0].discard_outputs = discard_outputs
            results = p_map(
                _parallel_solve,
                list(range(len(instances))),
                num_cpus=n_jobs,
                desc=label,
            )
            results = [r for r in results if r[0]]
            stats = []
            for (idx, (s, instance)) in enumerate(results):
                stats.append(s)
                instances[idx] = instance
            self._restore_miplearn_logger()
            return stats

    def fit(
        self,
        training_instances: List[Instance],
        n_jobs: int = 1,
    ) -> None:
        if len(training_instances) == 0:
            logger.warning("Empty list of training instances provided. Skipping.")
            return
        Component.fit_multiple(
            self.components,
            training_instances,
            n_jobs=n_jobs,
        )

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

    @staticmethod
    def _compute_gap(ub: Optional[float], lb: Optional[float]) -> Optional[float]:
        if lb is None or ub is None or lb * ub < 0:
            # solver did not find a solution and/or bound
            return None
        elif abs(ub - lb) < 1e-6:
            # avoid division by zero when ub = lb = 0
            return 0.0
        else:
            # divide by max(abs(ub),abs(lb)) to ensure gap <= 1
            return (ub - lb) / max(abs(ub), abs(lb))
